//! Robustness fuzz: malformed / truncated / corrupted input must make the decoder return an
//! `Err` (or cleanly stop), never panic. The decoder trusts conformant input for *correctness*,
//! but a panic on a bad file is a denial of service, so this bounds that surface.
//!
//! Deterministic (a seeded xorshift, std-only -- no fuzzing crate). It feeds, through both the
//! full `IvfReader` -> `Decoder::decode_frame` pipeline and directly into `decode_frame` (a
//! single mutated VP9 frame), each inside `catch_unwind`:
//!
//! 1. pure random byte buffers (always runs -- exercises the container/header entry points),
//! 2. every-prefix truncations of valid seed streams,
//! 3. random single/multi-bit corruptions of valid seed streams and their first frame.
//!
//! The seed-based passes skip cleanly when the vectors aren't downloaded; the random pass always
//! runs. Deepen coverage with `VP9DEC_FUZZ_ITERS=<n>`.

mod common;

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use vp9dec::ivf::IvfReader;
use vp9dec::test_support::BitWriter;
use vp9dec::{BoolCoderError, DecodeError, Decoder, DecoderLimits, HeaderError, TileError};

/// Source location (`file:line:col`) of the most recent panic, recorded by the fuzz's panic hook
/// -- a caught panic payload carries only the message, not the location that pinpoints the fix.
static PANIC_LOC: Mutex<Option<String>> = Mutex::new(None);

/// Per-attempt decoded-frame cap. The interesting decode paths (key + a couple of inter frames)
/// are all reached within a handful of frames, so this keeps each fuzz attempt cheap and also caps
/// the outer loop against a corrupted frame-count/size field that would otherwise decode a movie.
const FRAME_CAP: usize = 4;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
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

/// Runs `f` and returns the panic payload as a string if it unwound, `Ok(())` otherwise. The
/// caller silences the panic hook once around the whole run (below), so this doesn't touch it.
fn caught<F: FnOnce()>(f: F) -> Result<(), String> {
    *PANIC_LOC.lock().unwrap() = None;
    panic::catch_unwind(AssertUnwindSafe(f)).map_err(|p| {
        let msg = p
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".to_string());
        let loc = PANIC_LOC.lock().unwrap().take().unwrap_or_default();
        format!("{loc} | {msg}")
    })
}

/// The whole container -> decode pipeline on one IVF byte buffer. Any normal outcome (decoded, a
/// `DecodeError`, or an unreadable IVF) is `Ok`; only a panic is `Err`.
fn run_pipeline(bytes: &[u8]) -> Result<(), String> {
    caught(|| {
        let Ok(reader) = IvfReader::new(bytes) else {
            return;
        };
        let mut dec = Decoder::new();
        for (i, frame) in reader.enumerate() {
            if i >= FRAME_CAP {
                break;
            }
            let Ok(frame) = frame else {
                break;
            };
            let _ = dec.decode_frame(frame.data);
        }
    })
}

/// One raw byte buffer straight into `decode_frame` (a single, possibly-mutated VP9 frame chunk --
/// reaches the bitstream decoder without going through the IVF container).
fn run_frame(chunk: &[u8]) -> Result<(), String> {
    caught(|| {
        let _ = Decoder::new().decode_frame(chunk);
    })
}

fn rec(failures: &mut Vec<String>, label: String, res: Result<(), String>) {
    if let Err(msg) = res {
        eprintln!("[PANIC] {label}: {msg}");
        failures.push(format!("{label}: {msg}"));
    }
}

/// Small, diverse seeds: intra-only + superframe, segmentation, 10-bit (profile 2), 4:4:4
/// (profile 1), a 4-tile-column clip so mutations also reach the tile-PARALLEL decode path
/// (`tile::decode_tiles_parallel`, engaged only for >1 tile column) -- corrupting a multi-tile
/// stream must not panic on any worker thread either -- and an inter-frame-resize clip so
/// mutations also reach the reference-SCALED prediction path (steps != 16: the scaled AVX2
/// kernel, its scalar fallback, and the per-block reference-size bound
/// `TileError::RefFrameSizeOutOfRange`, none of which a fixed-resolution seed can exercise).
/// The 2 MB subpixel clip is deliberately omitted (its size makes the corruption pass' clones
/// dominate the runtime for no extra coverage).
const SEED_VECTORS: [&str; 6] = [
    "vp90-2-16-intra-only.ivf",
    "vp90-2-15-segkey.ivf",
    "vp92-2-20-10bit-yuv420.ivf",
    "vp91-2-04-yuv444.ivf",
    "vp90-2-08-tile_1x4.ivf",
    "vp90-2-21-resize_inter_320x180_5_1-2.ivf",
];

fn corrupt(rng: &mut Rng, buf: &mut [u8]) {
    if buf.is_empty() {
        return;
    }
    let k = 1 + rng.below(8);
    for _ in 0..k {
        let p = rng.below(buf.len());
        buf[p] ^= 1 << rng.below(8);
    }
}

#[test]
fn malformed_input_never_panics() {
    // `VP9DEC_FUZZ_ITERS` cranks the depth for a deliberate deep run; otherwise a light default.
    let env_iters = std::env::var("VP9DEC_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok());
    let iters = env_iters.unwrap_or(if cfg!(debug_assertions) { 32 } else { 200 });
    // The seed-based passes decode (perturbed) real frames -- ~10x slower in debug, too slow for a
    // routine debug `cargo test`. They run in release, or in debug only when iters is explicitly
    // requested (a deliberate deep / overflow-catching run). The cheap random-buffer pass, which
    // fails fast at container/header parse, always runs.
    let run_seeds = !cfg!(debug_assertions) || env_iters.is_some();

    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let mut failures: Vec<String> = Vec::new();

    // Silence the panic hook for the whole fuzz: a caught panic is an expected outcome here, and
    // the default hook would print a backtrace for every one. catch_unwind still gets the payload.
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        *PANIC_LOC.lock().unwrap() = Some(loc);
    }));

    // 1. Pure random buffers (always runs).
    for i in 0..iters {
        let len = rng.below(4096);
        let buf: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
        rec(
            &mut failures,
            format!("random-ivf[{i}] len={len}"),
            run_pipeline(&buf),
        );
        rec(
            &mut failures,
            format!("random-frame[{i}] len={len}"),
            run_frame(&buf),
        );
    }

    // 2 & 3. Truncations and corruptions of valid seeds (skip-clean if absent).
    let mut seeds_used = 0;
    for name in SEED_VECTORS {
        if !run_seeds {
            break;
        }
        let Some(bytes) = common::read_vector(name) else {
            continue;
        };
        seeds_used += 1;

        let first_frame: Option<Vec<u8>> = IvfReader::new(&bytes)
            .ok()
            .and_then(|mut r| r.next())
            .and_then(|f| f.ok())
            .map(|f| f.data.to_vec());

        // 2. Truncation: every prefix over the first 64 bytes (IVF + first frame header
        // boundaries), then 128 evenly-spaced prefixes across the rest -- bounded regardless of
        // seed size.
        let head = bytes.len().min(64);
        for len in 0..head {
            rec(
                &mut failures,
                format!("{name} trunc@{len}"),
                run_pipeline(&bytes[..len]),
            );
        }
        for t in 0..128 {
            let len = head + (bytes.len() - head) * t / 128;
            rec(
                &mut failures,
                format!("{name} trunc@{len}"),
                run_pipeline(&bytes[..len]),
            );
        }

        // 3. Random corruptions of the whole IVF and of the first frame.
        for i in 0..iters {
            let mut m = bytes.clone();
            corrupt(&mut rng, &mut m);
            rec(
                &mut failures,
                format!("{name} corrupt-ivf[{i}]"),
                run_pipeline(&m),
            );

            if let Some(fr) = &first_frame {
                let mut mf = fr.clone();
                corrupt(&mut rng, &mut mf);
                rec(
                    &mut failures,
                    format!("{name} corrupt-frame[{i}]"),
                    run_frame(&mf),
                );
            }
        }
    }

    panic::set_hook(prev_hook);

    eprintln!(
        "[fuzz] iters={iters} seeds_used={}/{} -> {} panic(s)",
        seeds_used,
        SEED_VECTORS.len(),
        failures.len()
    );
    if seeds_used == 0 {
        eprintln!(
            "[fuzz] seed passes did not run ({}); only the random-buffer pass ran",
            if run_seeds {
                "vectors not fetched -- see scripts/fetch-vectors.*"
            } else {
                "debug build -- run in release, or set VP9DEC_FUZZ_ITERS to force the deep run"
            }
        );
    }
    assert!(
        failures.is_empty(),
        "decoder panicked on {} malformed input(s); first {} shown:\n{}",
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

// =========================================================================
// Targeted regression tests for stateful error recovery, DPB edge cases,
// trailing zero padding, and resource limits (Fable review V-1..V-6).
// =========================================================================

fn write_header_tail(
    w: &mut BitWriter,
    width: u32,
    seg_enabled_no_update: bool,
    tile_cols_log2: u32,
    comp_len: usize,
) {
    // loop_filter_params (level=0, sharpness=0, delta_enabled=0)
    w.push_bits(0, 6);
    w.push_bits(0, 3);
    w.push_flag(false);
    // quantization_params (base_q_idx=60, no deltas)
    w.push_bits(60, 8);
    w.push_flag(false);
    w.push_flag(false);
    w.push_flag(false);
    // segmentation_params
    if seg_enabled_no_update {
        w.push_flag(true); // enabled
        w.push_flag(false); // update_map = 0
        w.push_flag(false); // update_data = 0
    } else {
        w.push_flag(false);
    }
    // tile_info
    let sb64_cols = width.div_ceil(64);
    let mut min_log2 = 0u32;
    while (64u32 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 -= 1;
    assert!((min_log2..=max_log2).contains(&tile_cols_log2));
    for _ in min_log2..tile_cols_log2 {
        w.push_flag(true);
    }
    if tile_cols_log2 < max_log2 {
        w.push_flag(false);
    }
    w.push_flag(false); // tile_rows_log2 = 0
    w.push_bits(comp_len as u32, 16);
}

fn synth_keyframe(width: u32, height: u32, tile_cols_log2: u32, tile: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_bits(2, 2); // frame_marker
    w.push_bits(0, 2); // profile 0
    w.push_flag(false); // show_existing_frame
    w.push_bits(0, 1); // KEY_FRAME
    w.push_flag(true); // show_frame
    w.push_flag(false); // error_resilient_mode
    w.push_bits(0x49, 8);
    w.push_bits(0x83, 8);
    w.push_bits(0x42, 8);
    w.push_bits(1, 3); // CS_BT_601
    w.push_flag(false); // color_range
    w.push_bits(width - 1, 16);
    w.push_bits(height - 1, 16);
    w.push_flag(false); // render_size
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
    let comp = [0u8];
    write_header_tail(&mut w, width, false, tile_cols_log2, comp.len());
    let mut out = w.finish();
    out.extend_from_slice(&comp);
    out.extend_from_slice(tile);
    out
}

fn synth_inter_frame(
    size: Option<(u32, u32)>,
    seg_enabled_no_update: bool,
    comp: &[u8],
    tile: &[u8],
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_bits(2, 2); // frame_marker
    w.push_bits(0, 2); // profile 0
    w.push_flag(false); // show_existing_frame
    w.push_bits(1, 1); // NON_KEY_FRAME
    w.push_flag(true); // show_frame
    w.push_flag(false); // error_resilient_mode
    w.push_bits(0, 2); // reset_frame_context
    w.push_bits(0xFF, 8); // refresh_frame_flags
    for _ in 0..3 {
        w.push_bits(0, 3); // ref_frame_idx = 0
        w.push_flag(false); // sign_bias
    }
    let width = match size {
        None => {
            w.push_flag(true); // found_ref = 1 on ref_frame_idx[0]
            64
        }
        Some((wd, ht)) => {
            w.push_flag(false);
            w.push_flag(false);
            w.push_flag(false);
            w.push_bits(wd - 1, 16);
            w.push_bits(ht - 1, 16);
            wd
        }
    };
    w.push_flag(false); // render_size
    w.push_flag(false); // allow_high_precision_mv
    w.push_flag(true); // is_filter_switchable
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
    write_header_tail(&mut w, width, seg_enabled_no_update, 0, comp.len());
    let mut out = w.finish();
    out.extend_from_slice(comp);
    out.extend_from_slice(tile);
    out
}

fn synth_profile1_444_intra_only(width: u32, height: u32, comp: &[u8], tile: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_bits(2, 2); // frame_marker
    w.push_bits(1, 1); // profile_low_bit = 1
    w.push_bits(0, 1); // profile_high_bit = 0 -> profile 1
    w.push_flag(false); // show_existing_frame
    w.push_bits(1, 1); // NON_KEY_FRAME
    w.push_flag(false); // show_frame = 0 (required for intra_only)
    w.push_flag(false); // error_resilient_mode
    w.push_flag(true); // intra_only = 1
    w.push_bits(0, 2); // reset_frame_context
    w.push_bits(0x49, 8);
    w.push_bits(0x83, 8);
    w.push_bits(0x42, 8);
    w.push_bits(1, 3); // CS_BT_601
    w.push_flag(false); // color_range
    w.push_bits(0, 1); // subsampling_x = 0 (4:4:4)
    w.push_bits(0, 1); // subsampling_y = 0 (4:4:4)
    w.push_bits(0, 1); // reserved_zero
    w.push_bits(0x00, 8); // refresh_frame_flags
    w.push_bits(width - 1, 16);
    w.push_bits(height - 1, 16);
    w.push_flag(false); // render_size
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
    write_header_tail(&mut w, width, false, 0, comp.len());
    let mut out = w.finish();
    out.extend_from_slice(comp);
    out.extend_from_slice(tile);
    out
}

fn minimal_valid_64x64_zero_tile_len() -> usize {
    for n in 1..600 {
        let mut d = Decoder::new();
        if d.decode_frame(&synth_keyframe(64, 64, 0, &vec![0u8; n]))
            .is_ok()
        {
            return n;
        }
    }
    panic!("no valid all-zero tile size found for 64x64 keyframe");
}

#[test]
fn prev_segment_ids_stays_consistent_after_failed_resized_frame() {
    // Regression test for V-1:
    // Frame A: 64x64 keyframe (Ok)
    // Frame B: 8x8 inter frame failing in compressed header (and a second variant failing in tile)
    // Frame C: 64x64 inter frame with segmentation enabled, update_map=0 (must not panic)
    let tile_len = minimal_valid_64x64_zero_tile_len();
    let kf = synth_keyframe(64, 64, 0, &vec![0u8; tile_len]);
    let seg_inter = synth_inter_frame(None, true, &[0], &vec![0u8; tile_len]);

    // Variant 1: Frame B fails at compressed header marker bit.
    let mut d = Decoder::new();
    d.decode_frame(&kf).expect("frame A (64x64 keyframe)");
    let bad_ch = synth_inter_frame(Some((8, 8)), false, &[0x80], &[0]);
    assert!(matches!(
        d.decode_frame(&bad_ch),
        Err(DecodeError::CompressedHeader(_))
    ));
    let _ = d.decode_frame(&seg_inter);

    // Variant 2: Frame B passes compressed header and fails in tile decode.
    let mut d2 = Decoder::new();
    d2.decode_frame(&kf).expect("frame A (64x64 keyframe)");
    let bad_tile = synth_inter_frame(Some((8, 8)), false, &[0], &[0x80]);
    assert!(matches!(
        d2.decode_frame(&bad_tile),
        Err(DecodeError::Tile(_))
    ));
    let _ = d2.decode_frame(&seg_inter);
}

#[test]
fn empty_dpb_found_ref_and_zero_slot_inter_never_panic() {
    // Regression test for V-2:
    // (a) Inter frame with found_ref=1 on an empty DPB must return Err (never Ok 0x0 frame).
    let mut d = Decoder::new();
    let empty_found_ref = synth_inter_frame(None, false, &[0], &[0]);
    assert_eq!(
        d.decode_frame(&empty_found_ref),
        Err(DecodeError::Header(HeaderError::InvalidFrameSize))
    );

    // (b) Subsequent 64x64 inter frame referencing unwritten DPB slots must return Err cleanly
    // in both AVX2 SIMD and VP9DEC_NO_SIMD=1 scalar paths (never panic in clamp).
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    for _ in 0..40 {
        let mut tile: Vec<u8> = (0..200).map(|_| (rng.next_u64() >> 24) as u8).collect();
        tile[0] &= 0x7F;
        let inter = synth_inter_frame(Some((64, 64)), false, &[0], &tile);
        let res = d.decode_frame(&inter);
        assert!(res.is_err(), "inter frame with empty DPB must fail cleanly");
    }
}

#[test]
fn trailing_zero_padding_accepted_and_nonzero_garbage_rejected() {
    // Regression test for V-4: spec §9.2.3 permits arbitrary trailing zero padding, while
    // non-zero trailing garbage past TILE_UNDER_READ_LIMIT_BITS must still return CorruptTile.
    let n = minimal_valid_64x64_zero_tile_len();
    let baseline = Decoder::new()
        .decode_frame(&synth_keyframe(64, 64, 0, &vec![0u8; n]))
        .expect("baseline keyframe must decode");

    for extra in [64usize, 129, 200, 1000] {
        let padded = Decoder::new()
            .decode_frame(&synth_keyframe(64, 64, 0, &vec![0u8; n + extra]))
            .unwrap_or_else(|e| panic!("+{extra} trailing zero bytes should decode, got {e:?}"));
        assert_eq!(padded, baseline, "+{extra} trailing zeros changed output");
    }

    // Trailing non-zero garbage > 1024 bits must be rejected as CorruptTile.
    let mut garbage = vec![0u8; n + 200];
    garbage[n + 199] = 0x01;
    assert_eq!(
        Decoder::new().decode_frame(&synth_keyframe(64, 64, 0, &garbage)),
        Err(DecodeError::Tile(TileError::CorruptTile))
    );
}

#[test]
fn decoder_limits_and_pre_allocation_tile_checks() {
    // Regression test for V-5: custom limits and cheap pre-allocation rejections.
    let n = minimal_valid_64x64_zero_tile_len();
    let kf_64 = synth_keyframe(64, 64, 0, &vec![0u8; n]);

    let limits = DecoderLimits {
        max_luma_samples: 64 * 64,
        max_width: 64,
        max_height: 64,
    };
    let mut d = Decoder::with_limits(limits);
    d.decode_frame(&kf_64)
        .expect("64x64 fits within 64x64 limits");

    // Exceeds max_width.
    assert_eq!(
        d.decode_frame(&synth_keyframe(128, 64, 0, &[0])),
        Err(DecodeError::Header(HeaderError::FrameSizeTooLarge))
    );
    // Exceeds max_height.
    assert_eq!(
        d.decode_frame(&synth_keyframe(64, 128, 0, &[0])),
        Err(DecodeError::Header(HeaderError::FrameSizeTooLarge))
    );
    // Exceeds max_luma_samples with unbounded per-axis dimensions.
    let mut d_samples = Decoder::with_limits(DecoderLimits {
        max_luma_samples: 64 * 64,
        ..DecoderLimits::default()
    });
    assert_eq!(
        d_samples.decode_frame(&synth_keyframe(128, 64, 0, &[0])),
        Err(DecodeError::Header(HeaderError::FrameSizeTooLarge))
    );

    // Pre-allocation tile layout rejections on a large 4096x2048 header:
    let mut d_big = Decoder::new();
    // Empty single-tile payload.
    assert_eq!(
        d_big.decode_frame(&synth_keyframe(4096, 2048, 0, &[])),
        Err(DecodeError::Tile(TileError::BoolCoder(
            BoolCoderError::EmptyBuffer
        )))
    );
    // Bad marker bit on single-tile payload.
    assert_eq!(
        d_big.decode_frame(&synth_keyframe(4096, 2048, 0, &[0x80])),
        Err(DecodeError::Tile(TileError::BoolCoder(
            BoolCoderError::InvalidMarker
        )))
    );
    // 4-tile header (tile_cols_log2=2) with fewer bytes than the 3x4-byte size headers + 4 tiles.
    assert_eq!(
        d_big.decode_frame(&synth_keyframe(4096, 2048, 2, &[0u8; 5])),
        Err(DecodeError::Tile(TileError::InvalidTileSize))
    );
    // Valid marker 1-byte truncated tile on a large frame fails fast via periodic over-read check.
    assert_eq!(
        d_big.decode_frame(&synth_keyframe(4096, 2048, 0, &[0x00])),
        Err(DecodeError::Tile(TileError::CorruptTile))
    );
}

#[test]
fn failed_intra_only_does_not_poison_last_color_config() {
    // Regression test for V-6: a failing profile-1 4:4:4 intra-only frame must not overwrite
    // `last_color_config` and break subsequent valid profile-0 4:2:0 inter frames.
    let n = minimal_valid_64x64_zero_tile_len();
    let mut d = Decoder::new();
    d.decode_frame(&synth_keyframe(64, 64, 0, &vec![0u8; n]))
        .expect("initial 4:2:0 keyframe must decode");

    let corrupt_444 = synth_profile1_444_intra_only(64, 64, &[0x80], &[0]);
    assert!(matches!(
        d.decode_frame(&corrupt_444),
        Err(DecodeError::CompressedHeader(_))
    ));

    // A subsequent 4:2:0 inter frame must NOT fail with RefFrameFormatMismatch.
    let inter = synth_inter_frame(None, false, &[0], &vec![0u8; n]);
    assert_ne!(
        d.decode_frame(&inter),
        Err(DecodeError::Header(HeaderError::RefFrameFormatMismatch))
    );
}
