//! Cross-check the no-`.md5` `bbb` movie clips against ffmpeg per displayed frame (I420
//! byte-for-byte). These full-length Big Buck Bunny clips ship no upstream `.ivf.md5`, so the
//! official sweep can't MD5-check them; ffmpeg is the independent oracle here -- the same role it
//! plays for the 12 tos/sintel clips (which were cross-checked in full).
//!
//! This is a **bounded spot-check**: each clip is ~14-18k frames, so decoding all seven with both
//! decoders is a multi-hour job with little marginal value over the exhaustive tos/sintel result.
//! We compare only the first [`SPOT_CHECK_FRAMES`] displayed frames of each clip -- enough to
//! cover the key frame, several GOPs, each tile layout (1x1 / 1x2 / 1x4), and frame-parallel
//! mode. To re-verify exhaustively, raise [`SPOT_CHECK_FRAMES`] to `usize::MAX`.
//!
//! ffmpeg's `framemd5` hash equals the MD5 of the packed `yuv420p` frame (verified), i.e. the
//! same Y->U->V byte string [`common::i420_bytes`] builds, so the two are directly comparable.
//!
//! Release-only + skip-clean: needs the `bbb` vectors fetched and an ffmpeg with a VP9 decoder;
//! it skips cleanly (and passes) when either is absent. Enforcing run:
//! `cargo test --release --test bbb_cross_check -- --nocapture`.

mod common;

use std::path::Path;
use std::process::Command;

use common::md5::{md5, to_hex};
use vp9dec::ivf::IvfReader;
use vp9dec::Decoder;

/// How many leading displayed frames of each clip to cross-check. The clips run to ~14-18k
/// frames each; this bounds the run to minutes while still covering the key frame, multiple GOPs,
/// the tile layout, and frame-parallel mode.
const SPOT_CHECK_FRAMES: usize = 1000;

const BBB_CLIPS: [&str; 7] = [
    "vp90-2-bbb_426x240_tile_1x1_180kbps.ivf",
    "vp90-2-bbb_640x360_tile_1x2_337kbps.ivf",
    "vp90-2-bbb_854x480_tile_1x2_651kbps.ivf",
    "vp90-2-bbb_1280x720_tile_1x4_1310kbps.ivf",
    "vp90-2-bbb_1920x1080_tile_1x1_2581kbps.ivf",
    "vp90-2-bbb_1920x1080_tile_1x4_2586kbps.ivf",
    "vp90-2-bbb_1920x1080_tile_1x4_fpm_2304kbps.ivf",
];

/// Locates and probes an ffmpeg binary: `VP9DEC_FFMPEG` (a full path) if set, else `"ffmpeg"` on
/// `PATH` -- never a hardcoded machine-specific path. A set-but-unusable `VP9DEC_FFMPEG` fails
/// loudly rather than skipping (a silent skip on a set var would be invisible).
fn probe_ffmpeg() -> Option<String> {
    let explicit = std::env::var("VP9DEC_FFMPEG").ok();
    let ffmpeg = explicit.clone().unwrap_or_else(|| "ffmpeg".to_string());
    let found = Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .is_ok_and(|out| out.status.success());
    assert!(
        found || explicit.is_none(),
        "VP9DEC_FFMPEG is set ({ffmpeg:?}) but does not run as ffmpeg"
    );
    found.then_some(ffmpeg)
}

/// The first VP9 decoder this ffmpeg build provides, preferring the libvpx reference over
/// ffmpeg's native one. Token-exact match on the decoder-name column ("vp9" is a substring of
/// "libvpx-vp9"). `None` if neither is present.
fn vp9_decoder(ffmpeg: &str) -> Option<&'static str> {
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-decoders"])
        .output()
        .expect("run ffmpeg -decoders");
    let listing = String::from_utf8_lossy(&out.stdout);
    let has = |name: &str| {
        listing
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(name))
    };
    ["libvpx-vp9", "vp9"].into_iter().find(|d| has(d))
}

/// vp9dec's lowercase I420 MD5 for each of the first `cap` displayed frames of `ivf_bytes`.
fn vp9dec_frame_md5s(ivf_bytes: &[u8], cap: usize) -> Vec<String> {
    let reader = IvfReader::new(ivf_bytes).expect("parse IVF header");
    let mut decoder = Decoder::new();
    let mut out = Vec::new();
    for frame in reader {
        if out.len() >= cap {
            break;
        }
        let frame = frame.expect("read IVF frame");
        for df in decoder.decode_frame(frame.data).expect("decode frame") {
            if let Some(decoded) = df.frame {
                out.push(to_hex(&md5(&common::i420_bytes(&decoded))).to_ascii_lowercase());
                if out.len() >= cap {
                    break;
                }
            }
        }
    }
    out
}

/// ffmpeg's lowercase per-frame `framemd5` (== MD5 of the packed I420 frame) for the first `cap`
/// displayed frames of `ivf_path`, decoded with `decoder`.
fn ffmpeg_frame_md5s(ffmpeg: &str, decoder: &str, ivf_path: &Path, cap: usize) -> Vec<String> {
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-c:v", decoder, "-i"])
        .arg(ivf_path)
        .args(["-frames:v", &cap.to_string(), "-f", "framemd5", "-"])
        .output()
        .expect("run ffmpeg framemd5");
    assert!(
        out.status.success(),
        "ffmpeg framemd5 failed on {}: {}",
        ivf_path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            l.rsplit(',')
                .next()
                .expect("framemd5 line has a hash column")
                .trim()
                .to_ascii_lowercase()
        })
        .collect()
}

#[test]
fn bbb_clips_cross_decode_against_ffmpeg() {
    // Release-only: decoding thousands of HD frames in a debug build is ~10x slower.
    if cfg!(debug_assertions) {
        eprintln!("[skip] debug build -- run with `cargo test --release --test bbb_cross_check`");
        return;
    }

    let Some(ffmpeg) = probe_ffmpeg() else {
        eprintln!(
            "[skip] ffmpeg not found (set VP9DEC_FFMPEG=<path> or put ffmpeg on PATH to run the \
             bbb cross-check)"
        );
        return;
    };
    let Some(decoder) = vp9_decoder(&ffmpeg) else {
        eprintln!("[skip] {ffmpeg:?} provides neither the libvpx-vp9 nor the vp9 decoder");
        return;
    };

    let mut ran = 0usize;
    for name in BBB_CLIPS {
        let path = common::vectors_dir().join(name);
        if !path.exists() {
            eprintln!("[skip] {name} not fetched");
            continue;
        }
        let ivf_bytes = std::fs::read(&path).expect("read bbb ivf");

        let ours = vp9dec_frame_md5s(&ivf_bytes, SPOT_CHECK_FRAMES);
        // Cap ffmpeg to exactly what we produced (a clip shorter than the cap yields fewer).
        let theirs = ffmpeg_frame_md5s(&ffmpeg, decoder, &path, ours.len());

        assert_eq!(
            ours.len(),
            theirs.len(),
            "{name}: displayed-frame count differs (vp9dec {}, ffmpeg {})",
            ours.len(),
            theirs.len()
        );
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(a, b, "{name}: I420 MD5 mismatch at displayed frame {i}");
        }
        eprintln!(
            "[xcheck] {name}: first {} displayed frames byte-identical to ffmpeg ({decoder})",
            ours.len()
        );
        ran += 1;
    }

    if ran == 0 {
        eprintln!(
            "[skip] no bbb clips present -- fetch the corpus with scripts/fetch-vectors.{{sh,ps1}}"
        );
    }
}
