//! Structure-aware malformed-input fuzzing.
//!
//! `robustness_test.rs` deliberately destroys arbitrary bytes. This complementary test first
//! parses each pristine VP9 frame and then mutates only one of two entropy-coded regions:
//!
//! 1. the compressed header, leaving the complete uncompressed header and its declared size
//!    unchanged;
//! 2. one tile payload, leaving the uncompressed/compressed headers, tile count, every 4-byte
//!    tile-size prefix, IVF chunk boundaries, and any superframe index unchanged.
//!
//! Earlier chunks are decoded unchanged before a later target is mutated. Mutations can therefore
//! reach stateful inter prediction and other post-header paths instead of overwhelmingly stopping
//! at the frame marker, sync code, or container parser. Every mutation is length-preserving and
//! deterministic (fixed xorshift seed); `Err` and a clean decode are both valid, but a panic is
//! always a failure.
//!
//! The normal test uses a bounded synthetic corpus even when official vectors are absent. A
//! release build also adds diverse official seeds when they have been fetched. Run a concrete,
//! bounded long campaign with 10,000 additional randomized attempts like this:
//!
//! ```text
//! VP9DEC_FUZZ_LONG_ITERS=10000 cargo test --release --test structured_fuzz_test -- --nocapture
//! ```
//!
//! In PowerShell, set `$env:VP9DEC_FUZZ_LONG_ITERS = "10000"` before the same `cargo test`
//! command. The summary reports compressed-header and tile-payload attempt counts separately.

mod common;

use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use common::encoder::{
    assemble_frame, build_inter_compressed_header, build_inter_header,
    build_keyframe_compressed_header, build_keyframe_header, build_keyframe_header_with_config,
    encode_inter_tile_forced, encode_keyframe_tile, encode_tree, header_size, kb, KeyframeConfig,
    SegSpec, HEIGHT, WIDTH,
};
use vp9dec::header::{
    parse_uncompressed_header, FrameHeader, LoopFilterDeltas, PersistentState, SegFeaturePersist,
    SEG_LVL_REF_FRAME, SEG_LVL_SKIP,
};
use vp9dec::ivf::IvfReader;
use vp9dec::prob_tables::{
    DC_PRED, DEFAULT_SKIP_PROB, H_PRED, INTRA_MODE_TREE, KF_PARTITION_PROBS, KF_UV_MODE_PROBS,
    KF_Y_MODE_PROBS, LAST_FRAME, PARTITION_NONE, PARTITION_TREE, V_PRED,
};
use vp9dec::test_support::BoolEncoder;
use vp9dec::Decoder;

/// First panic source location in the current caught attempt. Decoder worker threads use the
/// same hook; retaining the first site prevents a later `thread::scope` propagation panic from
/// replacing the original decoder location.
static PANIC_LOC: Mutex<Option<String>> = Mutex::new(None);

type PanicHook = Box<dyn Fn(&panic::PanicHookInfo<'_>) + Send + Sync + 'static>;

struct PanicHookGuard(Option<PanicHook>);

impl PanicHookGuard {
    fn install(hook: PanicHook) -> Self {
        let previous = panic::take_hook();
        panic::set_hook(hook);
        Self(Some(previous))
    }

    /// Runs `f` behind an outer unwind boundary so this guard is always dropped while the
    /// thread is no longer panicking. (`panic::set_hook` itself may not be called from a
    /// panicking thread.)
    fn run<F, T>(hook: PanicHook, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let guard = Self::install(hook);
        let result = panic::catch_unwind(AssertUnwindSafe(f));
        drop(guard);
        match result {
            Ok(value) => value,
            Err(payload) => panic::resume_unwind(payload),
        }
    }
}

impl Drop for PanicHookGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            panic::set_hook(previous);
        }
    }
}

/// Decode at most this many IVF chunks from a large seed for each mutation.
const CHUNK_CAP: usize = 4;
/// A chunk may contain several superframe constituents; bound their region-discovery work too.
const CONSTITUENT_CAP: usize = 16;
const DEFAULT_DEBUG_ATTEMPTS: usize = 32;
const DEFAULT_RELEASE_ATTEMPTS: usize = 256;

const OFFICIAL_SEEDS: [&str; 6] = [
    "vp90-2-16-intra-only.ivf",
    "vp90-2-15-segkey.ivf",
    "vp92-2-20-10bit-yuv420.ivf",
    "vp91-2-04-yuv444.ivf",
    "vp90-2-08-tile_1x4.ivf",
    "vp90-2-21-resize_inter_320x180_5_1-2.ivf",
];

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegionKind {
    CompressedHeader,
    TilePayload,
}

impl RegionKind {
    fn label(self) -> &'static str {
        match self {
            Self::CompressedHeader => "compressed",
            Self::TilePayload => "tile",
        }
    }
}

#[derive(Clone, Debug)]
struct Region {
    chunk: usize,
    constituent: usize,
    bytes: Range<usize>,
    kind: RegionKind,
}

struct Seed {
    name: &'static str,
    chunks: Vec<Vec<u8>>,
    regions: Vec<Region>,
}

#[derive(Default)]
struct AttemptCounts {
    compressed: usize,
    tile: usize,
}

impl AttemptCounts {
    fn record(&mut self, kind: RegionKind) {
        match kind {
            RegionKind::CompressedHeader => self.compressed += 1,
            RegionKind::TilePayload => self.tile += 1,
        }
    }
}

fn caught<F, T>(f: F) -> Result<T, String>
where
    F: FnOnce() -> T,
{
    *PANIC_LOC.lock().unwrap() = None;
    panic::catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".to_string());
        let location = PANIC_LOC.lock().unwrap().take().unwrap_or_default();
        format!("{location} | {message}")
    })
}

fn decode_sequence(chunks: &[Vec<u8>]) -> Result<(), vp9dec::DecodeError> {
    let mut decoder = Decoder::new();
    for chunk in chunks {
        decoder.decode_frame(chunk)?;
    }
    Ok(())
}

fn update_persistent_state(state: &mut PersistentState, header: &vp9dec::header::NewFrameHeader) {
    for (slot, size) in state.ref_frame_sizes.iter_mut().enumerate() {
        if (header.refresh_frame_flags >> slot) & 1 == 1 {
            *size = (header.width, header.height);
        }
    }
    state.loop_filter_deltas = LoopFilterDeltas {
        ref_deltas: header.loop_filter.ref_deltas,
        mode_deltas: header.loop_filter.mode_deltas,
    };
    state.segmentation = SegFeaturePersist {
        enabled: header.segmentation.feature_enabled,
        data: header.segmentation.feature_data,
        abs_or_delta: header.segmentation.abs_or_delta_update,
    };
}

/// Returns tile payload byte ranges relative to one constituent frame. Size prefixes are
/// deliberately excluded so mutations cannot change the declared tile partitioning.
fn tile_payload_ranges(
    frame: &[u8],
    tile_data_start: usize,
    tile_count: usize,
) -> Option<Vec<Range<usize>>> {
    let mut cursor = tile_data_start;
    let mut ranges = Vec::with_capacity(tile_count);
    for tile in 0..tile_count {
        let size = if tile + 1 == tile_count {
            frame.len().checked_sub(cursor)?
        } else {
            let size_end = cursor.checked_add(4)?;
            let size_bytes: [u8; 4] = frame.get(cursor..size_end)?.try_into().ok()?;
            cursor = size_end;
            u32::from_be_bytes(size_bytes) as usize
        };
        let end = cursor.checked_add(size)?;
        frame.get(cursor..end)?;
        ranges.push(cursor..end);
        cursor = end;
    }
    Some(ranges)
}

fn discover_regions(chunks: &[Vec<u8>]) -> Vec<Region> {
    let mut state = PersistentState::default();
    let mut regions = Vec::new();
    let mut constituents_seen = 0usize;

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let mut frame_offset = 0usize;
        for frame in vp9dec::superframe::split_superframe(chunk) {
            if constituents_seen >= CONSTITUENT_CAP {
                return regions;
            }
            let constituent = constituents_seen;
            constituents_seen += 1;

            let Ok((parsed, consumed)) = parse_uncompressed_header(frame, &state) else {
                frame_offset += frame.len();
                continue;
            };
            let FrameHeader::New(header) = parsed else {
                frame_offset += frame.len();
                continue;
            };

            let compressed_start = consumed;
            let Some(compressed_end) =
                compressed_start.checked_add(header.header_size_in_bytes as usize)
            else {
                frame_offset += frame.len();
                continue;
            };
            if compressed_end > frame.len() {
                frame_offset += frame.len();
                continue;
            }

            // Keep the bool coder's marker byte intact. Mutating later entropy bytes preserves
            // the entire uncompressed header and reaches the compressed-header parser.
            if compressed_end.saturating_sub(compressed_start) > 1 {
                regions.push(Region {
                    chunk: chunk_index,
                    constituent,
                    bytes: (frame_offset + compressed_start + 1)..(frame_offset + compressed_end),
                    kind: RegionKind::CompressedHeader,
                });
            }

            let tile_cols = 1usize.checked_shl(header.tile_cols_log2);
            let tile_rows = 1usize.checked_shl(header.tile_rows_log2);
            if let (Some(tile_cols), Some(tile_rows)) = (tile_cols, tile_rows) {
                if let Some(tile_count) = tile_cols.checked_mul(tile_rows) {
                    if let Some(tile_ranges) =
                        tile_payload_ranges(frame, compressed_end, tile_count)
                    {
                        for tile_range in tile_ranges {
                            // Keep each tile bool coder's marker byte intact as well.
                            if tile_range.len() > 1 {
                                regions.push(Region {
                                    chunk: chunk_index,
                                    constituent,
                                    bytes: (frame_offset + tile_range.start + 1)
                                        ..(frame_offset + tile_range.end),
                                    kind: RegionKind::TilePayload,
                                });
                            }
                        }
                    }
                }
            }

            update_persistent_state(&mut state, &header);
            frame_offset += frame.len();
        }
    }
    regions
}

fn mutate_anchor(bytes: &mut [u8], region: &Range<usize>, ordinal: usize) {
    let len = region.len();
    let pos = region.start + (len * 3 / 4).min(len - 1);
    bytes[pos] ^= 1 << (ordinal & 7);
}

fn mutate_random(bytes: &mut [u8], region: &Range<usize>, rng: &mut Rng) {
    let target = &mut bytes[region.clone()];
    // Preserve at least the first quarter of this already marker-excluded entropy region. A
    // changed bit therefore appears after a valid structural prefix instead of at its entrance.
    let preserved = target.len() / 4;
    let mutable = &mut target[preserved..];
    let pos = rng.below(mutable.len());

    match rng.below(4) {
        0 => mutable[pos] ^= 1 << rng.below(8),
        1 => mutable[pos] ^= (rng.next_u64() as u8) | 1,
        2 => {
            let run = (1 + rng.below(8)).min(mutable.len() - pos);
            let mask = (rng.next_u64() as u8) | 1;
            for byte in &mut mutable[pos..pos + run] {
                *byte ^= mask;
            }
        }
        _ => {
            let other = rng.below(mutable.len());
            if other == pos || mutable[other] == mutable[pos] {
                mutable[pos] = !mutable[pos];
            } else {
                mutable.swap(pos, other);
            }
        }
    }
}

fn run_mutation(
    seed: &Seed,
    region: &Region,
    label: &str,
    mutate: impl FnOnce(&mut [u8], &Range<usize>),
    failures: &mut Vec<String>,
) {
    let mut chunks = seed.chunks[..=region.chunk].to_vec();
    mutate(&mut chunks[region.chunk], &region.bytes);
    if let Err(message) = caught(|| {
        // Any ordinary DecodeError is a valid malformed-input outcome.
        let _ = decode_sequence(&chunks);
    }) {
        let failure = format!(
            "{} {label} chunk={} constituent={} {}@{:?}: {message}",
            seed.name,
            region.chunk,
            region.constituent,
            region.kind.label(),
            region.bytes
        );
        eprintln!("[PANIC] {failure}");
        failures.push(failure);
    }
}

fn synthetic_stateful_chunks() -> Vec<Vec<u8>> {
    let key_compressed = build_keyframe_compressed_header();
    let key_header = build_keyframe_header(
        WIDTH,
        HEIGHT,
        0,
        false,
        &SegSpec::disabled(),
        header_size(&key_compressed),
    );
    let key_tile = encode_keyframe_tile(
        [
            kb(None, V_PRED),
            kb(None, V_PRED),
            kb(None, H_PRED),
            kb(None, H_PRED),
        ],
        [128; 7],
    );
    let key = assemble_frame(key_header, key_compressed, key_tile);

    let mut segmentation = SegSpec::enabled();
    segmentation.feature_enabled[0][SEG_LVL_SKIP] = true;
    segmentation.feature_enabled[0][SEG_LVL_REF_FRAME] = true;
    segmentation.feature_data[0][SEG_LVL_REF_FRAME] = LAST_FRAME as i32;
    let inter_compressed = build_inter_compressed_header();
    let inter_header = build_inter_header(
        [0, 0, 0],
        None,
        0,
        &segmentation,
        header_size(&inter_compressed),
    );
    let inter_tile = encode_inter_tile_forced([0; 4], segmentation.tree_probs);
    let inter = assemble_frame(inter_header, inter_compressed, inter_tile);

    vec![key, inter]
}

fn encode_uniform_intra_tile(mode: u8) -> Vec<u8> {
    let mut encoder = BoolEncoder::new();
    for superblock_col in 0..4 {
        encode_tree(&mut encoder, &PARTITION_TREE, PARTITION_NONE, |node| {
            KF_PARTITION_PROBS[12][node]
        });
        encoder.write_bool(true, DEFAULT_SKIP_PROB[usize::from(superblock_col != 0)]);
        let left_mode = if superblock_col == 0 { DC_PRED } else { mode };
        encode_tree(&mut encoder, &INTRA_MODE_TREE, mode, |node| {
            KF_Y_MODE_PROBS[DC_PRED as usize][left_mode as usize][node]
        });
        encode_tree(&mut encoder, &INTRA_MODE_TREE, mode, |node| {
            KF_UV_MODE_PROBS[mode as usize][node]
        });
    }
    encoder.finish()
}

fn synthetic_hbd_multitile_chunk() -> Vec<u8> {
    const HBD_WIDTH: u32 = 512;
    const HBD_HEIGHT: u32 = 64;

    let compressed = build_keyframe_compressed_header();
    let header = build_keyframe_header_with_config(
        HBD_WIDTH,
        HBD_HEIGHT,
        0,
        false,
        &SegSpec::disabled(),
        header_size(&compressed),
        KeyframeConfig {
            profile: 3,
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            tile_cols_log2: 1,
        },
    );
    let mut left = encode_uniform_intra_tile(V_PRED);
    let mut right = encode_uniform_intra_tile(H_PRED);
    left.extend_from_slice(&[0; 4]);
    right.extend_from_slice(&[0; 4]);

    let mut tile_data = Vec::with_capacity(4 + left.len() + right.len());
    tile_data.extend_from_slice(
        &u32::try_from(left.len())
            .expect("synthetic tile size fits u32")
            .to_be_bytes(),
    );
    tile_data.extend_from_slice(&left);
    tile_data.extend_from_slice(&right);
    assemble_frame(header, compressed, tile_data)
}

fn make_superframe(frames: &[Vec<u8>]) -> Vec<u8> {
    assert!(!frames.is_empty() && frames.len() <= 8);
    let largest = frames.iter().map(Vec::len).max().unwrap();
    let bytes_per_size = if largest <= u8::MAX as usize {
        1
    } else if largest <= u16::MAX as usize {
        2
    } else if largest <= 0x00ff_ffff {
        3
    } else {
        4
    };
    let marker = 0xc0 | (((bytes_per_size - 1) as u8) << 3) | (frames.len() as u8 - 1);

    let mut chunk = Vec::new();
    for frame in frames {
        chunk.extend_from_slice(frame);
    }
    chunk.push(marker);
    for frame in frames {
        let size = frame.len() as u32;
        chunk.extend_from_slice(&size.to_le_bytes()[..bytes_per_size]);
    }
    chunk.push(marker);
    chunk
}

fn load_official_seed(name: &'static str) -> Option<Seed> {
    let bytes = common::read_vector(name)?;
    let reader = IvfReader::new(&bytes)
        .unwrap_or_else(|error| panic!("{name}: valid seed has malformed IVF: {error:?}"));
    let chunks = reader
        .take(CHUNK_CAP)
        .map(|frame| {
            frame
                .unwrap_or_else(|error| panic!("{name}: malformed IVF frame: {error:?}"))
                .data
                .to_vec()
        })
        .collect::<Vec<_>>();
    assert!(!chunks.is_empty(), "{name}: valid seed contains no frames");
    let regions = discover_regions(&chunks);
    Some(Seed {
        name,
        chunks,
        regions,
    })
}

fn long_attempts() -> Option<usize> {
    let value = std::env::var("VP9DEC_FUZZ_LONG_ITERS").ok()?;
    let attempts = value.parse::<usize>().unwrap_or_else(|_| {
        panic!("VP9DEC_FUZZ_LONG_ITERS must be a positive integer, got {value:?}")
    });
    assert!(
        attempts > 0,
        "VP9DEC_FUZZ_LONG_ITERS must be greater than zero"
    );
    Some(attempts)
}

#[test]
fn structure_aware_mutations_never_panic() {
    let opt_in_attempts = long_attempts();
    let random_attempts = opt_in_attempts.unwrap_or(if cfg!(debug_assertions) {
        DEFAULT_DEBUG_ATTEMPTS
    } else {
        DEFAULT_RELEASE_ATTEMPTS
    });

    let stateful = synthetic_stateful_chunks();
    let superframe = make_superframe(&[stateful[0].clone(), stateful[0].clone()]);
    let mut seeds = vec![
        Seed {
            name: "synthetic-stateful-key-inter",
            regions: discover_regions(&stateful),
            chunks: stateful,
        },
        Seed {
            name: "synthetic-superframe",
            regions: discover_regions(std::slice::from_ref(&superframe)),
            chunks: vec![superframe],
        },
        {
            let chunks = vec![synthetic_hbd_multitile_chunk()];
            Seed {
                name: "synthetic-profile3-hbd-multitile",
                regions: discover_regions(&chunks),
                chunks,
            }
        },
    ];

    let run_official = !cfg!(debug_assertions) || opt_in_attempts.is_some();
    let mut official_used = 0usize;
    if run_official {
        for name in OFFICIAL_SEEDS {
            if let Some(seed) = load_official_seed(name) {
                official_used += 1;
                seeds.push(seed);
            }
        }
    }

    for seed in &seeds {
        assert!(
            !seed.regions.is_empty(),
            "{}: region discovery was vacuous",
            seed.name
        );
    }

    let (failures, attempts, region_count) = PanicHookGuard::run(
        Box::new(|info| {
            let location = info
                .location()
                .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
                .unwrap_or_default();
            let mut recorded = PANIC_LOC.lock().unwrap();
            if recorded.is_none() {
                *recorded = Some(location);
            }
        }),
        || {
            let mut failures = Vec::new();
            let mut active = vec![false; seeds.len()];
            for (index, seed) in seeds.iter().enumerate() {
                match caught(|| decode_sequence(&seed.chunks)) {
                    Ok(Ok(())) => active[index] = true,
                    Ok(Err(error)) => failures.push(format!(
                        "{} pristine seed returned unexpected error: {error:?}",
                        seed.name
                    )),
                    Err(message) => {
                        failures.push(format!("{} pristine seed panicked: {message}", seed.name))
                    }
                }
            }

            let cases = seeds
                .iter()
                .enumerate()
                .filter(|(seed_index, _)| active[*seed_index])
                .flat_map(|(seed_index, seed)| {
                    (0..seed.regions.len()).map(move |region_index| (seed_index, region_index))
                })
                .collect::<Vec<_>>();
            assert!(!cases.is_empty(), "no pristine structured seed decoded");

            let mut attempts = AttemptCounts::default();
            for (ordinal, &(seed_index, region_index)) in cases.iter().enumerate() {
                let seed = &seeds[seed_index];
                let region = &seed.regions[region_index];
                attempts.record(region.kind);
                run_mutation(
                    seed,
                    region,
                    &format!("anchor[{ordinal}]"),
                    |bytes, range| mutate_anchor(bytes, range, ordinal),
                    &mut failures,
                );
            }

            let mut rng = Rng::new(0xD1B5_4A32_D192_ED03);
            for iteration in 0..random_attempts {
                let (seed_index, region_index) = cases[rng.below(cases.len())];
                let seed = &seeds[seed_index];
                let region = &seed.regions[region_index];
                attempts.record(region.kind);
                run_mutation(
                    seed,
                    region,
                    &format!("random[{iteration}]"),
                    |bytes, range| mutate_random(bytes, range, &mut rng),
                    &mut failures,
                );
            }

            (failures, attempts, cases.len())
        },
    );

    eprintln!(
        "[structured-fuzz] seeds={} official={official_used}/{} regions={} \
         random_attempts={random_attempts} compressed_attempts={} tile_attempts={} panics={}",
        seeds.len(),
        OFFICIAL_SEEDS.len(),
        region_count,
        attempts.compressed,
        attempts.tile,
        failures.len()
    );
    if !run_official {
        eprintln!(
            "[structured-fuzz] debug default used synthetic seeds only; use --release or set \
             VP9DEC_FUZZ_LONG_ITERS to include fetched official seeds"
        );
    }
    assert!(
        attempts.compressed > 0 && attempts.tile > 0,
        "both structure-aware mutation classes must run"
    );
    assert!(
        failures.is_empty(),
        "{} structured fuzz failure(s); first {} shown:\n{}",
        failures.len(),
        failures.len().min(25),
        failures
            .iter()
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
