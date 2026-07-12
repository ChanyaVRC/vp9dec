//! Integration tests for inter-frame bitstream decoding (M3 first half).
//!
//! If `.ivf` files have been downloaded into `tests/vectors/`, this decodes every frame in
//! the file (key frames, inter frames, and droppable frames included) in order via
//! [`Decoder`], and verifies that `uncompressed_header` + `compressed_header` + mode info/MV/
//! residual tokens for every tile can be read through to completion without panicking.
//!
//! Pixel generation (motion compensation, subpixel interpolation) is still a stub pending
//! implementation in M3 second half, so pixel value correctness isn't verified here
//! (`decode_test.rs`/`conformance_test.rs` separately verify key frame pixel correctness).
//! Correctly reading the bitstream through to the end is ensured by being able to keep
//! reading subsequent frames in sequence (i.e. without the bool decoder's consumed position
//! drifting) — if the consumed position drifted even for a single frame, the next frame's
//! `uncompressed_header` would fail early on its `frame_marker`/`frame_sync_code` etc. checks.
//!
//! In environments without test vectors, the corresponding test is skipped via early
//! return + `eprintln!` (see README.md for how to obtain them).

use std::path::Path;

use vp9dec::ivf::IvfReader;
use vp9dec::Decoder;

fn check_vector(relative_path: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join(relative_path);

    if !path.exists() {
        eprintln!(
            "[skip] Test vector not found, skipping: {}\n\
             Please download it beforehand following the instructions in README.md.",
            path.display()
        );
        return;
    }

    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read test vector {}: {e}", path.display()));
    let reader = IvfReader::new(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse IVF header for {}: {e:?}", path.display()));

    let mut decoder = Decoder::new();
    let mut frame_count = 0usize;
    let mut decoded_count = 0usize;
    let mut hidden_count = 0usize;

    for (i, frame) in reader.enumerate() {
        let frame = frame
            .unwrap_or_else(|e| panic!("{}: failed to read IVF frame {i}: {e:?}", path.display()));
        match decoder.decode_frame(frame.data) {
            Ok(Some(f)) => {
                decoded_count += 1;
                // Minimal sanity check: plane size matches the expected value derived from frame.width/height.
                assert_eq!(
                    f.y.len(),
                    (f.width * f.height) as usize,
                    "{}: frame {i}: unexpected Y plane size",
                    path.display()
                );
            }
            Ok(None) => {
                hidden_count += 1;
            }
            Err(e) => panic!(
                "{}: frame {i} (of {} total so far) failed to decode: {e:?}",
                path.display(),
                frame_count + 1
            ),
        }
        frame_count += 1;
    }

    eprintln!(
        "[ok] {}: read through all {frame_count} IVF frames (shown frames: {decoded_count}, hidden frames: {hidden_count})",
        path.display()
    );
    assert!(
        frame_count > 1,
        "{}: only a single frame (doesn't exercise inter-frame decoding)",
        path.display()
    );
    assert!(
        decoded_count > 1,
        "{}: 1 or fewer newly decoded frames (may not include any inter frames)",
        path.display()
    );
}

#[test]
fn vp90_2_12_droppable_1_reads_all_frames_to_completion() {
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_reads_all_frames_to_completion() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}
