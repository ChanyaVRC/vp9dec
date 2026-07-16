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

use std::collections::BTreeSet;
use std::path::Path;

use vp9dec::header::{SEG_LVL_ALT_L, SEG_LVL_ALT_Q, SEG_LVL_MAX, SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use vp9dec::ivf::IvfReader;
use vp9dec::md5::{md5, to_hex};
use vp9dec::{Decoder, Frame, FrameDecodeInfo};

/// Returns the `Frame`'s I420 bytes concatenated in Y->U->V order (the layout expected by `.ivf.md5`).
fn i420_bytes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.y.len() + frame.u.len() + frame.v.len());
    out.extend_from_slice(&frame.y);
    out.extend_from_slice(&frame.u);
    out.extend_from_slice(&frame.v);
    out
}

/// M3 second half: verifies that **every displayed frame** exactly matches the corresponding
/// line in `.ivf.md5` (this won't pass unless motion compensation, probability adaptation,
/// the DPB, and the loop filter's cross-frame state are all correct — a far stricter test
/// than verifying a single key frame).
///
/// `.ivf.md5` has "1 line = 1 output (displayed) frame"; each `DecodedFrame` with
/// `frame: Some` consumes one line and is compared (hidden constituent frames with
/// `show_frame == 0` have no corresponding line in `.ivf.md5`, so they don't consume one).
fn check_all_frames(ivf_name: &str) {
    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors");
    let ivf_path = vectors_dir.join(ivf_name);
    let md5_path = vectors_dir.join(format!("{ivf_name}.md5"));

    if !ivf_path.exists() || !md5_path.exists() {
        eprintln!(
            "[skip] Test vector or .ivf.md5 not found, skipping: {} / {}\n\
             Please download them beforehand following the instructions in README.md.",
            ivf_path.display(),
            md5_path.display()
        );
        return;
    }

    let ivf_bytes = std::fs::read(&ivf_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ivf_path.display()));
    let md5_text = std::fs::read_to_string(&md5_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", md5_path.display()));
    let expected_lines: Vec<String> = md5_text
        .lines()
        .map(|line| {
            line.split_whitespace()
                .next()
                .unwrap_or_else(|| panic!("{}: contains a blank line", md5_path.display()))
                .to_ascii_lowercase()
        })
        .collect();

    let reader = IvfReader::new(&ivf_bytes).unwrap_or_else(|e| {
        panic!(
            "failed to parse IVF header for {}: {e:?}",
            ivf_path.display()
        )
    });

    let mut decoder = Decoder::new();
    let mut output_idx = 0usize;
    let mut mismatches: Vec<usize> = Vec::new();

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
        for decoded in outcome.into_iter().filter_map(|df| df.frame) {
            let actual_bytes = i420_bytes(&decoded);
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
}

#[test]
fn vp90_2_12_droppable_1_all_frames_match_official_md5() {
    check_all_frames("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_all_frames_match_official_md5() {
    check_all_frames("vp90-2-09-subpixel-00.ivf");
}

/// Same as [`check_all_frames`] (bit-exact MD5 check of every displayed frame), plus a
/// coverage check: `predicate` is evaluated against the `info` of *every* returned
/// `DecodedFrame` that carries one (including hidden constituent frames with
/// `show_frame == 0`, since segmentation/intra_only state isn't limited to displayed
/// frames), and at least one frame must satisfy it. This proves the vector actually
/// exercises the decode path `description` names, rather than merely producing correct
/// output on some other path.
///
/// Also `eprintln!`s a one-line coverage summary (the set of `reset_frame_context`
/// values seen, and the union of `seg_features_active` across the whole stream) so a
/// human reading the test log can tell exactly which `SEG_LVL_*` levels and which
/// `reset_frame_context` values this vector exercises.
fn check_all_frames_with_coverage(
    ivf_name: &str,
    predicate: impl Fn(&FrameDecodeInfo) -> bool,
    description: &str,
) {
    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors");
    let ivf_path = vectors_dir.join(ivf_name);
    let md5_path = vectors_dir.join(format!("{ivf_name}.md5"));

    if !ivf_path.exists() || !md5_path.exists() {
        eprintln!(
            "[skip] Test vector or .ivf.md5 not found, skipping: {} / {}\n\
             Please download them beforehand following the instructions in README.md.",
            ivf_path.display(),
            md5_path.display()
        );
        return;
    }

    let ivf_bytes = std::fs::read(&ivf_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ivf_path.display()));
    let md5_text = std::fs::read_to_string(&md5_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", md5_path.display()));
    let expected_lines: Vec<String> = md5_text
        .lines()
        .map(|line| {
            line.split_whitespace()
                .next()
                .unwrap_or_else(|| panic!("{}: contains a blank line", md5_path.display()))
                .to_ascii_lowercase()
        })
        .collect();

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
                let actual_bytes = i420_bytes(&decoded);
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
fn vp90_2_15_segkey_exercises_segmentation() {
    check_all_frames_with_coverage(
        "vp90-2-15-segkey.ivf",
        |info| info.segmentation_enabled,
        "segmentation_enabled == true on some decoded frame",
    );
}

#[test]
fn vp90_2_15_segkey_adpq_exercises_segmentation() {
    check_all_frames_with_coverage(
        "vp90-2-15-segkey_adpq.ivf",
        |info| info.segmentation_enabled,
        "segmentation_enabled == true on some decoded frame",
    );
}

#[test]
fn vp90_2_16_intra_only_exercises_intra_only_frame() {
    check_all_frames_with_coverage(
        "vp90-2-16-intra-only.ivf",
        |info| info.intra_only,
        "intra_only == true on some decoded frame",
    );
}
