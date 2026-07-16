//! Parse-layer probes against the two local official WebM test vectors, using the public
//! `Decoder` API and the lower-level parsing entry points directly. Merged (Wave 3 test-layer
//! consolidation, 2026-07-16) from the former `header_test.rs` (uncompressed header fields),
//! `compressed_header_test.rs` (compressed-header read-through), and `decode_test.rs` (decode
//! plausibility) -- each a parse-layer probe complementing the bit-exact MD5 checks in
//! `conformance_test.rs`.
//!
//! In environments without test vectors, the corresponding test is skipped via early
//! return + `eprintln!` (see README.md for how to obtain them).

mod common;

use vp9dec::compressed_header::{parse_compressed_header, FrameContext};
use vp9dec::header::{parse_uncompressed_header, FrameHeader, FrameType, PersistentState};
use vp9dec::tile::TileDecoder;
use vp9dec::{Decoder, Frame};

// ===========================================================================================
// From header_test.rs: the IVF container parses correctly, the first frame is a key frame, and
// the uncompressed header's width/height match the IVF container header's width/height.
// ===========================================================================================

fn check_header_vector(relative_path: &str) {
    let Some(bytes) = common::read_vector(relative_path) else {
        return;
    };
    let path = common::vectors_dir().join(relative_path);
    let (ivf_header, first_frame_data) = common::first_ivf_frame(&bytes);
    assert_eq!(&ivf_header.fourcc, b"VP90", "codec fourcc should be VP90");

    let (parsed, _consumed) =
        parse_uncompressed_header(first_frame_data, &PersistentState::default()).unwrap_or_else(
            |e| {
                panic!(
                    "failed to parse uncompressed header of first frame in {}: {e:?}",
                    path.display()
                )
            },
        );

    match parsed {
        FrameHeader::New(f) => {
            assert_eq!(
                f.frame_type,
                FrameType::KeyFrame,
                "first frame of {} should be a key frame",
                path.display()
            );
            assert_eq!(
                f.width,
                ivf_header.width as u32,
                "decoded width should match IVF container header for {}",
                path.display()
            );
            assert_eq!(
                f.height,
                ivf_header.height as u32,
                "decoded height should match IVF container header for {}",
                path.display()
            );
            eprintln!(
                "[ok] {}: {}x{}, frame_type={:?}",
                path.display(),
                f.width,
                f.height,
                f.frame_type
            );
        }
        FrameHeader::ShowExistingFrame { .. } => {
            panic!(
                "first frame of {} unexpectedly used show_existing_frame",
                path.display()
            );
        }
    }
}

#[test]
fn vp90_2_12_droppable_1() {
    check_header_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00() {
    check_header_vector("vp90-2-09-subpixel-00.ivf");
}

// ===========================================================================================
// From compressed_header_test.rs: for the first key frame, "uncompressed_header ->
// compressed_header reads through to completion without panicking" and the parsed tx_mode /
// skip_prob fall within valid ranges.
//
// `decode_tiles` is now fully implemented including token decoding and reconstruction, but this
// test still primarily verifies that `compressed_header` reads through to completion, and only
// confirms that the call to `TileDecoder::decode_tiles` doesn't panic (either an `Ok` or `Err`
// result is accepted). Full pixel output correctness (a statistical sanity check) is verified by
// `check_decode_vector` below. Detailed correctness of tile/partition/mode info is verified by
// unit tests using synthetic bitstreams inside `src/tile.rs`.
// ===========================================================================================

fn check_compressed_header_vector(relative_path: &str) {
    let Some(bytes) = common::read_vector(relative_path) else {
        return;
    };
    let path = common::vectors_dir().join(relative_path);
    let (_ivf_header, first_frame_data) = common::first_ivf_frame(&bytes);

    let (parsed, consumed) =
        parse_uncompressed_header(first_frame_data, &PersistentState::default()).unwrap_or_else(
            |e| {
                panic!(
                    "failed to parse uncompressed header of first frame in {}: {e:?}",
                    path.display()
                )
            },
        );

    let header = match parsed {
        FrameHeader::New(h) => h,
        FrameHeader::ShowExistingFrame { .. } => {
            panic!(
                "first frame of {} unexpectedly used show_existing_frame",
                path.display()
            )
        }
    };

    let header_size = header.header_size_in_bytes as usize;
    assert!(
        header_size > 0,
        "{}: header_size_in_bytes should be non-zero for a real key frame",
        path.display()
    );
    let compressed_start = consumed;
    let compressed_end = compressed_start + header_size;
    assert!(
        compressed_end <= first_frame_data.len(),
        "{}: compressed header ({compressed_start}..{compressed_end}) exceeds frame data length {}",
        path.display(),
        first_frame_data.len()
    );
    let compressed_bytes = &first_frame_data[compressed_start..compressed_end];

    let compressed = parse_compressed_header(compressed_bytes, &header, FrameContext::default())
        .unwrap_or_else(|e| {
            panic!(
                "{}: failed to parse compressed_header (size={header_size}): {e:?}",
                path.display()
            )
        });

    // tx_mode falls within the range ONLY_4X4(0) to TX_MODE_SELECT(4).
    assert!(
        compressed.tx_mode <= 4,
        "{}: tx_mode out of range: {}",
        path.display(),
        compressed.tx_mode
    );
    // Per spec, for lossless frames tx_mode must always be ONLY_4X4 (0).
    if header.quantization.lossless {
        assert_eq!(
            compressed.tx_mode,
            0,
            "{}: lossless frame should force tx_mode = ONLY_4X4",
            path.display()
        );
    }
    // Per spec, probability values fall within 1..=255 (never 0, due to how
    // read_prob/diff_update_prob work).
    for &p in compressed.probs.skip_prob.iter() {
        assert!(
            p >= 1,
            "{}: skip_prob should never be 0, got {p}",
            path.display()
        );
    }

    eprintln!(
        "[ok] {}: header_size_in_bytes={header_size}, tx_mode={}, skip_prob={:?}",
        path.display(),
        compressed.tx_mode,
        compressed.probs.skip_prob
    );

    // Attempt decode_tiles on the tile data (from right after the compressed header to the
    // end of the frame data). As noted in the module doc, this test only confirms the call
    // doesn't panic (either Ok or Err is fine); full pixel/statistical correctness is checked
    // elsewhere.
    let tile_data = &first_frame_data[compressed_end..];
    let color_config = header.color_config.expect(
        "first frame of an IVF stream is always a key frame, which always parses color_config",
    );
    let mut tile_decoder = TileDecoder::new(&header, color_config, &compressed);
    match tile_decoder.decode_tiles(tile_data) {
        Ok(()) => eprintln!("[ok] {}: decode_tiles completed fully", path.display()),
        Err(e) => eprintln!("[info] {}: decode_tiles stopped with {e:?}", path.display()),
    }
}

#[test]
fn vp90_2_12_droppable_1_compressed_header() {
    check_compressed_header_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_compressed_header() {
    check_compressed_header_vector("vp90-2-09-subpixel-00.ivf");
}

// ===========================================================================================
// From decode_test.rs: decoding the first key frame via the public `Decoder` API, verifying
// that it decodes through to completion and that the resulting Y plane isn't uniform (i.e. it
// has statistics consistent with a real-world video vector).
// ===========================================================================================

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

fn check_decode_vector(relative_path: &str, expected_width: u32, expected_height: u32) {
    let Some(bytes) = common::read_vector(relative_path) else {
        return;
    };
    let path = common::vectors_dir().join(relative_path);
    let (_ivf_header, first_frame_data) = common::first_ivf_frame(&bytes);

    let mut decoder = Decoder::new();
    let frame = decoder
        .decode_frame(first_frame_data)
        .unwrap_or_else(|e| {
            panic!(
                "{}: decode_frame failed on first frame: {e:?}",
                path.display()
            )
        })
        .into_iter()
        .find_map(|df| df.frame)
        .unwrap_or_else(|| {
            panic!(
                "{}: first chunk produced no displayed frame (expected a shown key frame)",
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
    check_decode_vector("vp90-2-12-droppable_1.ivf", 352, 288);
}

#[test]
fn vp90_2_09_subpixel_00_decodes_first_keyframe() {
    check_decode_vector("vp90-2-09-subpixel-00.ivf", 320, 180);
}
