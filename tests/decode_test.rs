//! Integration tests for `decode_keyframe` (the M2 public API).
//!
//! Using real data downloaded into `tests/vectors/` (a VP9 stream containing a key frame),
//! this verifies that the first key frame decodes through to completion and that the
//! resulting Y plane isn't uniform (i.e. it has statistics consistent with a real-world
//! video vector).
//!
//! In environments without test vectors, the corresponding test is skipped via early
//! return + `eprintln!` (see README.md for how to obtain them).

use std::path::Path;

use vp9dec::ivf::IvfReader;
use vp9dec::{decode_keyframe, Frame};

struct YStats {
    min: u8,
    max: u8,
    mean: f64,
    variance: f64,
    all_same: bool,
}

fn y_stats(frame: &Frame) -> YStats {
    let min = *frame.y.iter().min().expect("non-empty Y plane");
    let max = *frame.y.iter().max().expect("non-empty Y plane");
    let n = frame.y.len() as f64;
    let mean = frame.y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = frame
        .y
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    YStats {
        min,
        max,
        mean,
        variance,
        all_same: min == max,
    }
}

fn check_vector(relative_path: &str, expected_width: u32, expected_height: u32) {
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
    let mut reader = IvfReader::new(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse IVF header for {}: {e:?}", path.display()));
    let first_frame = reader
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let frame = decode_keyframe(first_frame.data).unwrap_or_else(|e| {
        panic!(
            "{}: decode_keyframe failed on first frame: {e:?}",
            path.display()
        )
    });

    assert_eq!(
        frame.width,
        expected_width,
        "{}: unexpected width",
        path.display()
    );
    assert_eq!(
        frame.height,
        expected_height,
        "{}: unexpected height",
        path.display()
    );
    assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);

    let uv_w = (frame.width as usize).div_ceil(2);
    let uv_h = (frame.height as usize).div_ceil(2);
    assert_eq!(frame.u.len(), uv_w * uv_h);
    assert_eq!(frame.v.len(), uv_w * uv_h);

    let stats = y_stats(&frame);
    eprintln!(
        "[ok] {}: {}x{}, Y min={} max={} mean={:.2} variance={:.2}",
        path.display(),
        frame.width,
        frame.height,
        stats.min,
        stats.max,
        stats.mean,
        stats.variance
    );

    assert!(
        !stats.all_same,
        "{}: Y plane has a single uniform value across all pixels (decode result looks wrong)",
        path.display()
    );
    assert!(
        stats.variance > 0.0,
        "{}: Y plane variance is 0",
        path.display()
    );
    assert!(
        stats.min < 50 && stats.max > 200,
        "{}: luma range looks unnatural for a real-world video vector (min={}, max={})",
        path.display(),
        stats.min,
        stats.max
    );
}

#[test]
fn vp90_2_12_droppable_1_decodes_first_keyframe() {
    check_vector("vp90-2-12-droppable_1.ivf", 352, 288);
}

#[test]
fn vp90_2_09_subpixel_00_decodes_first_keyframe() {
    check_vector("vp90-2-09-subpixel-00.ivf", 320, 180);
}
