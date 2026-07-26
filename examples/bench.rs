//! Benchmark harness (SIMD wave 1 measurement infrastructure, see
//! docs/implementation-notes.md "SIMD wave 1"): decodes `.ivf` files and reports wall-clock
//! decode speed in megapixels/second (MP/s = sum of displayed-frame width*height / time),
//! per-clip and aggregated, with multiple iterations to damp noise (min + median reported).
//!
//! Must be run `--release` -- a debug build's decode speed is not representative (roughly
//! an order of magnitude slower) and would make the numbers meaningless.
//!
//! ```sh
//! # Default representative set (see DEFAULT_CLIPS below): a 1920-width clip, a ~854-width
//! # clip, and a small (426-width) clip, all already local under tests/vectors/.
//! cargo run --release --example bench
//!
//! # Specific files instead.
//! cargo run --release --example bench -- path/to/a.ivf path/to/b.ivf
//!
//! # Override the iteration count (default 3). Useful to lower for a large/slow clip --
//! # the 1920-width representative clip is a 214MB/17620-frame decode per iteration.
//! cargo run --release --example bench -- --iters=1 path/to/big.ivf
//!
//! # Per-stage timing breakdown (needs the bench-timing feature -- see src/bench_timing.rs;
//! # without it the stage numbers are all-zero). Runs one extra decode pass per file.
//! cargo run --release --features bench-timing --example bench -- --stages [files...]
//!
//! # Reproduce a tile-worker limit for dispatch benchmarks (needs test-support).
//! cargo run --release --features test-support --example bench -- --tile-workers=1 [files...]
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use vp9dec::ivf::IvfReader;
use vp9dec::Decoder;

const DEFAULT_ITERS: usize = 3;

/// The representative set named by the SIMD wave 1 mission: a 1920-width tos clip, a
/// ~854-width tos clip, and a small (426-width) bbb clip.
const DEFAULT_CLIPS: [&str; 3] = [
    "vp90-2-tos_1920x800_tile_1x4_fpm_2335kbps.ivf",
    "vp90-2-tos_854x356_tile_1x2_656kbps.ivf",
    "vp90-2-bbb_426x240_tile_1x1_180kbps.ivf",
];

/// Stage names in the exact order of `vp9dec::bench_timing::Stage`'s variants (matches the
/// index into `bench_timing::snapshot()`'s array).
const STAGE_NAMES: [&str; vp9dec::bench_timing::STAGE_COUNT] = [
    "Total",
    "HeaderParse",
    "CompressedHeader",
    "TileDecode",
    "LoopFilter",
    "DpbOutput",
    "  (in TileDecode) TokenDequantTransform",
    "  (in TileDecode) InterPredict",
    "  (in TileDecode) IntraPredict",
    "    (subset of TokenDequantTransform) InverseTransform",
];

fn main() {
    let mut show_stages = false;
    let mut iters = DEFAULT_ITERS;
    // Stop after this many DISPLAYED frames per clip (None = whole clip). The full-length
    // tos/sintel movies are 17k+ frames per iteration; for a stage-proportion profile a few
    // hundred frames of the same content is plenty and finishes in seconds.
    let mut max_frames: Option<u64> = None;
    let mut tile_workers: Option<usize> = None;
    let mut files: Vec<PathBuf> = Vec::new();

    for arg in std::env::args().skip(1) {
        if arg == "--stages" {
            show_stages = true;
        } else if let Some(n) = arg.strip_prefix("--iters=") {
            iters = n
                .parse()
                .unwrap_or_else(|_| panic!("--iters=N: N must be a positive integer, got {n:?}"));
        } else if let Some(n) = arg.strip_prefix("--max-frames=") {
            max_frames = Some(
                n.parse()
                    .unwrap_or_else(|_| panic!("--max-frames=N: N must be an integer, got {n:?}")),
            );
        } else if let Some(n) = arg.strip_prefix("--tile-workers=") {
            let n = n.parse().unwrap_or_else(|_| {
                panic!("--tile-workers=N: N must be a positive integer, got {n:?}")
            });
            assert!(n > 0, "--tile-workers=N: N must be positive");
            tile_workers = Some(n);
        } else {
            files.push(PathBuf::from(arg));
        }
    }
    if let Some(n) = tile_workers {
        #[cfg(feature = "test-support")]
        vp9dec::tile::FORCE_TILE_WORKERS.store(n, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(feature = "test-support"))]
        panic!("--tile-workers={n} requires --features test-support");
    }
    if files.is_empty() {
        let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("vectors");
        files = DEFAULT_CLIPS
            .iter()
            .map(|name| vectors_dir.join(name))
            .collect();
    }

    println!(
        "bench-timing feature: {}",
        if cfg!(feature = "bench-timing") {
            "on"
        } else {
            "off (--stages numbers would be all-zero; rebuild with --features bench-timing)"
        }
    );
    println!("iterations per clip: {iters}\n");
    if let Some(n) = tile_workers {
        println!("forced tile workers: {n}\n");
    }

    let mut total_megapixels = 0.0f64;
    let mut total_min_secs = 0.0f64;

    for path in &files {
        if !path.exists() {
            eprintln!("[skip] not found: {}", path.display());
            continue;
        }
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

        let mut durations = Vec::with_capacity(iters);
        let mut megapixels = 0.0f64;
        let mut frame_count = 0u64;
        let mut dims = (0u32, 0u32);
        for i in 0..iters {
            let result = decode_once(&bytes, max_frames);
            durations.push(result.wall_time);
            if i == 0 {
                megapixels = result.megapixels;
                frame_count = result.frame_count;
                dims = result.first_frame_dims;
            }
        }
        durations.sort();
        let min = durations[0];
        let median = durations[durations.len() / 2];

        println!(
            "{:<52} {:>5}x{:<5} frames={:<6} min={:>8.3}s median={:>8.3}s  MP/s(min)={:>7.2} MP/s(median)={:>7.2}",
            path.file_name().unwrap().to_string_lossy(),
            dims.0,
            dims.1,
            frame_count,
            min.as_secs_f64(),
            median.as_secs_f64(),
            megapixels / min.as_secs_f64(),
            megapixels / median.as_secs_f64(),
        );

        total_megapixels += megapixels;
        total_min_secs += min.as_secs_f64();

        if show_stages {
            print_stage_breakdown(&bytes, max_frames);
        }
    }

    if total_min_secs > 0.0 {
        println!(
            "\naggregate: {:.2} MP/s (sum of per-clip megapixels / sum of per-clip min times)",
            total_megapixels / total_min_secs
        );
    }
}

struct DecodeResult {
    wall_time: Duration,
    megapixels: f64,
    frame_count: u64,
    /// Dimensions of the first displayed frame (informational only; MP/s itself sums the
    /// actual per-frame width*height so a resolution change mid-clip wouldn't skew it).
    first_frame_dims: (u32, u32),
}

/// Decodes `ivf_bytes` once with a fresh [`Decoder`], stopping after `max_frames` displayed
/// frames if set (whole clip otherwise). Timing covers only the frames actually decoded.
fn decode_once(ivf_bytes: &[u8], max_frames: Option<u64>) -> DecodeResult {
    let reader = IvfReader::new(ivf_bytes).expect("failed to parse IVF header");
    let mut decoder = Decoder::new();
    let mut megapixels = 0.0f64;
    let mut frame_count = 0u64;
    let mut first_frame_dims = (0u32, 0u32);

    let start = Instant::now();
    'outer: for frame in reader {
        let frame = frame.expect("failed to read IVF frame");
        let decoded = decoder
            .decode_frame(frame.data)
            .expect("decode_frame failed");
        for df in decoded {
            if let Some(f) = df.frame {
                if first_frame_dims == (0, 0) {
                    first_frame_dims = (f.width, f.height);
                }
                megapixels += (f.width as f64 * f.height as f64) / 1_000_000.0;
                frame_count += 1;
                if max_frames.is_some_and(|m| frame_count >= m) {
                    break 'outer;
                }
            }
        }
    }
    DecodeResult {
        wall_time: start.elapsed(),
        megapixels,
        frame_count,
        first_frame_dims,
    }
}

/// Runs one extra decode pass with the stage counters reset first, then prints the
/// per-stage ms/percentage breakdown (see `src/bench_timing.rs`).
fn print_stage_breakdown(ivf_bytes: &[u8], max_frames: Option<u64>) {
    vp9dec::bench_timing::reset();
    let _ = decode_once(ivf_bytes, max_frames);
    let snap = vp9dec::bench_timing::snapshot();
    let total_ns = snap[0] as f64;

    println!("  -- stage breakdown --");
    println!(
        "  note: indented TileDecode stages sum worker elapsed time; their Total-wall % may exceed 100%"
    );
    for (name, ns) in STAGE_NAMES.iter().zip(snap.iter()) {
        let ms = *ns as f64 / 1e6;
        let pct = if total_ns > 0.0 {
            *ns as f64 / total_ns * 100.0
        } else {
            0.0
        };
        println!("  {name:<40} {ms:>10.1} ms  {pct:>5.1}%");
    }
    // The 5 coarse stages (indices 1..=5) don't cover everything charged to Total: notably
    // TileDecoder::new_with_prev's per-frame plane (re)allocation runs between the
    // CompressedHeader and TileDecode timers. Reporting this gap rather than silently
    // dropping it keeps the percentages honestly accounted for.
    let coarse_sum: u64 = snap[1..=5].iter().sum();
    let gap = snap[0].saturating_sub(coarse_sum);
    let gap_pct = if total_ns > 0.0 {
        gap as f64 / total_ns * 100.0
    } else {
        0.0
    };
    println!(
        "  {:<40} {:>10.1} ms  {:>5.1}%  (untimed: TileDecoder setup, superframe split, ...)",
        "other/gap (Total - 5 coarse stages)",
        gap as f64 / 1e6,
        gap_pct
    );
}
