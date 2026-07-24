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
use vp9dec::Decoder;

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
