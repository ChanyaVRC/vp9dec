//! Integration tests for compressed header (`compressed_header`) parsing, using official
//! WebM test vectors.
//!
//! If `.ivf` files have been downloaded into `tests/vectors/`, this verifies, for the first
//! key frame, that "uncompressed_header -> compressed_header reads through to completion
//! without panicking" and that the parsed tx_mode / skip_prob fall within valid ranges.
//!
//! `decode_tiles` is now fully implemented including token decoding and reconstruction, but
//! this test still primarily verifies that `compressed_header` reads through to completion,
//! and only confirms that the call to `TileDecoder::decode_tiles` doesn't panic (either an
//! `Ok` or `Err` result is accepted). Full pixel output correctness (a statistical sanity
//! check) is verified by the `decode_keyframe`-based tests in `tests/decode_test.rs`.
//! Detailed correctness of tile/partition/mode info is verified by unit tests using
//! synthetic bitstreams inside `src/tile.rs`.
//!
//! In environments without test vectors, the corresponding test is skipped via early
//! return + `eprintln!` (see README.md for how to obtain them).

use std::path::Path;

use vp9dec::compressed_header::parse_compressed_header;
use vp9dec::header::{
    parse_uncompressed_header, FrameHeader, MAX_SEGMENTS, NUM_REF_FRAMES, SEG_LVL_MAX,
};
use vp9dec::ivf::IvfReader;
use vp9dec::tile::TileDecoder;

const NO_REF_SIZES: [(u32, u32); NUM_REF_FRAMES] = [(0, 0); NUM_REF_FRAMES];
const NO_LF_DELTAS: ([i8; 4], [i8; 2]) = ([1, 0, -1, -1], [0, 0]);
const NO_SEG_FEATURES: ([[bool; SEG_LVL_MAX]; MAX_SEGMENTS], [[i32; SEG_LVL_MAX]; MAX_SEGMENTS], bool) =
    (
        [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
        [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        false,
    );

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

    let mut frames = reader;
    let first_frame = frames
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let (parsed, consumed) =
        parse_uncompressed_header(first_frame.data, &NO_REF_SIZES, NO_LF_DELTAS, NO_SEG_FEATURES)
            .unwrap_or_else(
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
        compressed_end <= first_frame.data.len(),
        "{}: compressed header ({compressed_start}..{compressed_end}) exceeds frame data length {}",
        path.display(),
        first_frame.data.len()
    );
    let compressed_bytes = &first_frame.data[compressed_start..compressed_end];

    let compressed = parse_compressed_header(compressed_bytes, header.quantization.lossless)
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
    let tile_data = &first_frame.data[compressed_end..];
    let mut tile_decoder = TileDecoder::new(&header, &compressed);
    match tile_decoder.decode_tiles(tile_data) {
        Ok(()) => eprintln!("[ok] {}: decode_tiles completed fully", path.display()),
        Err(e) => eprintln!("[info] {}: decode_tiles stopped with {e:?}", path.display()),
    }
}

#[test]
fn vp90_2_12_droppable_1_compressed_header() {
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_compressed_header() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}
