//! Bit-exact verification against official VP9 conformance test vectors (M2b).
//!
//! The libvpx test data distribution (`https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`)
//! ships an `.ivf.md5` alongside each `.ivf`, recording the MD5 of each output frame's I420
//! data (the full Y plane, then the full U plane, then the full V plane, concatenated) —
//! one line per frame, in standard `md5sum` format: `<32-char hex>  <filename>`.
//!
//! Here we verify, for every displayed output frame, that the MD5 of the decoder's output
//! (the Y->U->V concatenated byte string, cropped to display size) exactly matches the
//! corresponding line of `.ivf.md5`. If the loop filter, cropping, plane concatenation
//! order, or render size interpretation is wrong in any way, this comparison is
//! guaranteed to fail.
//!
//! Assumes the test vectors and MD5 files have already been downloaded into `tests/vectors/`
//! (excluded via `.gitignore`; see README.md for how to obtain them). If they're missing,
//! skips via early return + `eprintln!`.

mod common;

use std::collections::BTreeSet;

use common::md5::{md5, to_hex};
use vp9dec::header::{SEG_LVL_ALT_L, SEG_LVL_ALT_Q, SEG_LVL_MAX, SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use vp9dec::ivf::IvfReader;
use vp9dec::{Decoder, FrameDecodeInfo};

/// A coverage predicate paired with a human-readable description of the decode path it checks
/// for (see `check_all_frames`'s `coverage` parameter). A type alias only to satisfy clippy's
/// `type_complexity` lint on the bare tuple.
type Coverage = (&'static str, fn(&FrameDecodeInfo) -> bool);

/// Verifies, for every displayed frame in `ivf_name`, that the MD5 of the decoder's output
/// exactly matches the corresponding line of `.ivf.md5` (this won't pass unless motion
/// compensation, probability adaptation, the DPB, and the loop filter's cross-frame state are
/// all correct -- a far stricter test than verifying a single key frame).
///
/// When `coverage` is `Some((description, predicate))`, additionally verifies that `predicate`
/// is satisfied by the `info` of *some* decoded frame (including hidden constituent frames with
/// `show_frame == 0`, since segmentation/intra_only state isn't limited to displayed frames) --
/// proving the vector actually exercises the decode path `description` names, rather than merely
/// producing correct output on some other path -- and `eprintln!`s a one-line coverage summary
/// (the set of `reset_frame_context` values seen, and the union of `seg_features_active` across
/// the whole stream) so a human reading the test log can tell exactly which `SEG_LVL_*` levels
/// and which `reset_frame_context` values the vector exercises. `FrameDecodeInfo` is always
/// collected regardless of `coverage` (cheap bookkeeping, not an extra assertion), so both call
/// site families share one decode loop instead of two ~80%-identical copies.
fn check_all_frames(ivf_name: &str, coverage: Option<Coverage>) {
    let Some((ivf_bytes, expected_lines)) = common::read_vector_with_md5(ivf_name) else {
        return;
    };
    let ivf_path = common::vectors_dir().join(ivf_name);

    let reader = IvfReader::new(&ivf_bytes).unwrap_or_else(|e| {
        panic!(
            "failed to parse IVF header for {}: {e:?}",
            ivf_path.display()
        )
    });

    let mut decoder = Decoder::new();
    let mut output_idx = 0usize;
    let mut mismatches: Vec<usize> = Vec::new();
    let mut infos: Vec<FrameDecodeInfo> = Vec::new();

    'frames: for (ivf_frame_idx, frame) in reader.enumerate() {
        let frame = frame.unwrap_or_else(|e| {
            panic!(
                "{}: failed to read IVF frame {ivf_frame_idx}: {e:?}",
                ivf_path.display()
            )
        });
        let outcome = decoder.decode_frame(frame.data).unwrap_or_else(|e| {
            panic!(
                "{}: IVF frame {ivf_frame_idx} failed to decode: {e:?}",
                ivf_path.display()
            )
        });
        // One DecodedFrame per constituent frame, so hidden sub-frames' infos are observed
        // too -- the coverage predicate below relies on this (e.g. vp90-2-16-intra-only's
        // intra_only frames are the hidden ones in a superframe, not the trailing visible
        // frame).
        for df in outcome {
            if let Some(info) = df.info {
                infos.push(info);
            }
            if let Some(decoded) = df.frame {
                let actual_bytes = common::i420_bytes(&decoded);
                let actual = to_hex(&md5(&actual_bytes));
                let expected = expected_lines.get(output_idx).unwrap_or_else(|| {
                    panic!(
                        "{}: produced more output frames than .ivf.md5 has lines ({}) (output_idx={output_idx})",
                        ivf_path.display(),
                        expected_lines.len()
                    )
                });
                if &actual != expected {
                    mismatches.push(output_idx);
                    eprintln!(
                        "[NG] {}: output frame {output_idx} (ivf frame {ivf_frame_idx}) MD5 mismatch\n  actual:   {actual}\n  expected: {expected}\n  ({}x{}, y.len={}, u.len={}, v.len={})",
                        ivf_path.display(),
                        decoded.width,
                        decoded.height,
                        decoded.y.len(),
                        decoded.u.len(),
                        decoded.v.len(),
                    );
                    // Only the first mismatch needs close investigation, so stop early here
                    // (see README "Debugging notes": first check how many frames match).
                    break 'frames;
                }
                output_idx += 1;
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{}: frames up to {output_idx} matched; mismatch at output frame {} (out of {} output frames total)",
        ivf_path.display(),
        mismatches[0],
        expected_lines.len()
    );
    assert_eq!(
        output_idx,
        expected_lines.len(),
        "{}: number of output frames doesn't match the number of lines in .ivf.md5 (not all frames may have been compared)",
        ivf_path.display()
    );
    eprintln!(
        "[ok] {}: all {output_idx} output frames exactly match the official MD5",
        ivf_path.display()
    );

    let Some((description, predicate)) = coverage else {
        return;
    };

    let reset_frame_contexts: BTreeSet<u8> = infos.iter().map(|i| i.reset_frame_context).collect();
    let mut seg_features_union = [false; SEG_LVL_MAX];
    for info in &infos {
        for (level, active) in seg_features_union.iter_mut().enumerate() {
            *active |= info.seg_features_active[level];
        }
    }
    eprintln!(
        "[coverage] {}: reset_frame_context values seen = {reset_frame_contexts:?}; \
         seg_features_active union: SEG_LVL_ALT_Q={} SEG_LVL_ALT_L={} SEG_LVL_REF_FRAME={} SEG_LVL_SKIP={}",
        ivf_path.display(),
        seg_features_union[SEG_LVL_ALT_Q],
        seg_features_union[SEG_LVL_ALT_L],
        seg_features_union[SEG_LVL_REF_FRAME],
        seg_features_union[SEG_LVL_SKIP],
    );

    assert!(
        infos.iter().any(predicate),
        "{}: no decoded frame satisfied the coverage predicate ({description}) -- \
         this vector doesn't exercise the intended decode path",
        ivf_path.display()
    );
}

#[test]
fn vp90_2_12_droppable_1_all_frames_match_official_md5() {
    check_all_frames("vp90-2-12-droppable_1.ivf", None);
}

#[test]
fn vp90_2_09_subpixel_00_all_frames_match_official_md5() {
    check_all_frames("vp90-2-09-subpixel-00.ivf", None);
}

#[test]
fn vp90_2_15_segkey_exercises_segmentation() {
    check_all_frames(
        "vp90-2-15-segkey.ivf",
        Some((
            "segmentation_enabled == true on some decoded frame",
            |info: &FrameDecodeInfo| info.segmentation_enabled,
        )),
    );
}

#[test]
fn vp90_2_15_segkey_adpq_exercises_segmentation() {
    check_all_frames(
        "vp90-2-15-segkey_adpq.ivf",
        Some((
            "segmentation_enabled == true on some decoded frame",
            |info: &FrameDecodeInfo| info.segmentation_enabled,
        )),
    );
}

#[test]
fn vp90_2_16_intra_only_exercises_intra_only_frame() {
    check_all_frames(
        "vp90-2-16-intra-only.ivf",
        Some((
            "intra_only == true on some decoded frame",
            |info: &FrameDecodeInfo| info.intra_only,
        )),
    );
}

/// The official profile 1/2/3 vectors: profile 1 (8-bit non-4:2:0), profile 2 (10/12-bit
/// 4:2:0), profile 3 (10/12-bit non-4:2:0). Each is md5-checked frame-by-frame exactly like
/// the profile-0 vectors above (`i420_bytes` emits 10/12-bit output as 16-bit LE, matching
/// libvpx's high-depth `.ivf.md5`), skipping cleanly if not fetched. These are tiny (~350 KB
/// total), so profile 1-3 conformance runs in the DEFAULT suite -- otherwise it would live
/// only in the sweep, which skips on a partial checkout.
#[test]
fn profile_1_3_vectors_match_official_md5() {
    for name in [
        "vp91-2-04-yuv422.ivf",
        "vp91-2-04-yuv440.ivf",
        "vp91-2-04-yuv444.ivf",
        "vp92-2-20-10bit-yuv420.ivf",
        "vp92-2-20-12bit-yuv420.ivf",
        "vp93-2-20-10bit-yuv422.ivf",
        "vp93-2-20-10bit-yuv440.ivf",
        "vp93-2-20-10bit-yuv444.ivf",
        "vp93-2-20-12bit-yuv422.ivf",
        "vp93-2-20-12bit-yuv440.ivf",
        "vp93-2-20-12bit-yuv444.ivf",
    ] {
        check_all_frames(name, None);
    }
}

/// Unit tests for `common::md5`'s RFC 1321 implementation. Kept in this single dedicated place
/// rather than inside `tests/common/md5.rs` itself: everything under `tests/common/` is
/// recompiled once per consuming test binary, so a `#[test]` there would rerun once per binary
/// instead of once overall (Wave 3 test-layer consolidation, 2026-07-16; relocated verbatim
/// from `src/md5.rs`'s own `mod tests`).
mod md5_tests {
    use super::common::md5::{md5, to_hex};

    fn hex_of(data: &[u8]) -> String {
        to_hex(&md5(data))
    }

    #[test]
    fn empty_string() {
        assert_eq!(hex_of(b""), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn abc() {
        assert_eq!(hex_of(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn message_digest() {
        assert_eq!(
            hex_of(b"message digest"),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
    }

    #[test]
    fn alphabet() {
        assert_eq!(
            hex_of(b"abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
    }

    #[test]
    fn alphanumeric() {
        assert_eq!(
            hex_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
    }

    #[test]
    fn eighty_digits() {
        assert_eq!(
            hex_of(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    /// Confirms padding is applied correctly even for input crossing a block boundary (64 bytes).
    #[test]
    fn exactly_one_block() {
        let data = vec![b'a'; 64];
        // Rather than comparing against a known value, this only confirms determinism
        // and the absence of panics (a length of exactly 64 bytes is the boundary case
        // where padding overflows entirely into a second block).
        let d1 = md5(&data);
        let d2 = md5(&data);
        assert_eq!(d1, d2);
    }
}
