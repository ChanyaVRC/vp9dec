//! Integration tests using official WebM test vectors (libvpx conformance test data).
//!
//! If `.ivf` files have been downloaded into `tests/vectors/`, this verifies:
//! - The IVF container parses correctly
//! - The first frame is a key frame
//! - The uncompressed header's width/height match the IVF container header's width/height
//!
//! So that the test suite as a whole doesn't fail in environments without test vectors
//! (e.g. CI without network access), a test is skipped via early return + `eprintln!` when
//! the file isn't found. See README.md for how to obtain the vectors.

use std::path::Path;

use vp9dec::header::{parse_uncompressed_header, FrameHeader, FrameType, PersistentState};
use vp9dec::ivf::IvfReader;

/// Verifies, for the given test vector, that "the IVF can be read / the first frame is a
/// key frame / the header's width and height match the IVF header".
/// Returns early if the file doesn't exist.
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
    let ivf_header = reader.header().clone();
    assert_eq!(&ivf_header.fourcc, b"VP90", "codec fourcc should be VP90");

    let mut frames = reader;
    let first_frame = frames
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let (parsed, _consumed) =
        parse_uncompressed_header(first_frame.data, &PersistentState::default()).unwrap_or_else(
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
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}
