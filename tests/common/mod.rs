//! Shared test infrastructure for the integration tests under `tests/` (Wave 3 test-layer
//! consolidation, 2026-07-16): vector/`.ivf.md5` loading, the skip-if-absent convention, and
//! I420 byte layout, extracted from what was previously duplicated across
//! `conformance_test.rs`/`header_test.rs`/`compressed_header_test.rs`/`decode_test.rs`/
//! `inter_frame_test.rs`.
//!
//! Each `tests/*.rs` file is compiled as its own crate, and each declares its own `mod common;`
//! pointing at this directory -- so this module is recompiled once per consuming test binary.
//! Not every binary uses every helper here (e.g. `api_test.rs` never touches `i420_bytes` or
//! `md5`), so `dead_code` is blanket-allowed for the whole module tree rather than per binary.
#![allow(dead_code)]

pub mod encoder;
pub mod md5;

use std::path::{Path, PathBuf};

use vp9dec::ivf::{IvfHeader, IvfReader};
use vp9dec::{Frame, PlaneData};

/// `tests/vectors/`, where downloaded conformance vectors live (see README.md).
pub fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
}

/// Reads `tests/vectors/<relative_path>`. Returns `None` (after an `eprintln!`) if the file is
/// missing, so callers can `return` immediately -- tests must SKIP cleanly, not fail, when
/// vectors haven't been downloaded (see README.md).
pub fn read_vector(relative_path: &str) -> Option<Vec<u8>> {
    let path = vectors_dir().join(relative_path);
    if !path.exists() {
        eprintln!(
            "[skip] Test vector not found, skipping: {}\n\
             Please download it beforehand following the instructions in README.md.",
            path.display()
        );
        return None;
    }
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display())))
}

/// One lowercased MD5 hex digest per non-blank line of an `.ivf.md5` file's text (each
/// line's first whitespace-separated column; a filename column, if present, is discarded),
/// in the file's order -- one line per displayed output frame.
pub fn parse_md5_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Appends `.<ext>` to a path's full file name -- the sidecar naming `fetch-vectors` uses
/// (`foo.ivf` -> `foo.ivf.md5`/`foo.ivf.res`). Not [`Path::with_extension`], which replaces
/// after the last dot -- appending is correct regardless of how many dots are in the
/// vector's own name (e.g. `...b6-.v2.ivf`).
pub fn sidecar_path(path: &Path, ext: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".");
    name.push(ext);
    PathBuf::from(name)
}

/// Reads `tests/vectors/<ivf_name>` together with its accompanying `<ivf_name>.md5`. Returns
/// `None` (after an `eprintln!`) if either file is missing. The returned `Vec<String>` is
/// [`parse_md5_lines`] of the `.ivf.md5` file.
pub fn read_vector_with_md5(ivf_name: &str) -> Option<(Vec<u8>, Vec<String>)> {
    let ivf_path = vectors_dir().join(ivf_name);
    let md5_path = vectors_dir().join(format!("{ivf_name}.md5"));

    if !ivf_path.exists() || !md5_path.exists() {
        eprintln!(
            "[skip] Test vector or .ivf.md5 not found, skipping: {} / {}\n\
             Please download them beforehand following the instructions in README.md.",
            ivf_path.display(),
            md5_path.display()
        );
        return None;
    }

    let ivf_bytes = std::fs::read(&ivf_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ivf_path.display()));
    let md5_text = std::fs::read_to_string(&md5_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", md5_path.display()));

    Some((ivf_bytes, parse_md5_lines(&md5_text)))
}

/// Parses `bytes` as an IVF file and returns its container header plus the first frame's raw
/// data. Panics (not skip) on a malformed IVF -- absence has already been filtered out by
/// `read_vector`'s existence check before this ever runs.
pub fn first_ivf_frame(bytes: &[u8]) -> (IvfHeader, &[u8]) {
    let reader =
        IvfReader::new(bytes).unwrap_or_else(|e| panic!("failed to parse IVF header: {e:?}"));
    let ivf_header = reader.header().clone();
    let mut frames = reader;
    let first_frame = frames
        .next()
        .unwrap_or_else(|| panic!("IVF contains no frames"))
        .unwrap_or_else(|e| panic!("failed to read first frame: {e:?}"));
    (ivf_header, first_frame.data)
}

/// Appends one plane's samples to `out` in the layout libvpx's `.ivf.md5` uses: `U8` samples
/// as-is, `U16` samples (10/12-bit) as 2 little-endian bytes each (libvpx's high-bit-depth
/// raw/MD5 output is 16-bit LE).
fn push_plane_bytes(out: &mut Vec<u8>, data: &PlaneData) {
    match data {
        PlaneData::U8(v) => out.extend_from_slice(v),
        PlaneData::U16(v) => {
            for &sample in v {
                out.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }
}

/// The `Frame`'s I420 bytes concatenated in Y->U->V order (the layout `.ivf.md5` files use).
pub fn i420_bytes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::new();
    push_plane_bytes(&mut out, &frame.y);
    push_plane_bytes(&mut out, &frame.u);
    push_plane_bytes(&mut out, &frame.v);
    out
}
