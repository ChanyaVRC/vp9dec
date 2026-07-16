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
use vp9dec::Frame;

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

/// Reads `tests/vectors/<ivf_name>` together with its accompanying `<ivf_name>.md5`. Returns
/// `None` (after an `eprintln!`) if either file is missing. The returned `Vec<String>` is one
/// lowercased MD5 hex digest per line of `.ivf.md5` (its filename column, if present, is
/// discarded), in the file's order -- one line per displayed output frame.
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
    let expected_lines = md5_text
        .lines()
        .map(|line| {
            line.split_whitespace()
                .next()
                .unwrap_or_else(|| panic!("{}: contains a blank line", md5_path.display()))
                .to_ascii_lowercase()
        })
        .collect();

    Some((ivf_bytes, expected_lines))
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

/// The `Frame`'s I420 bytes concatenated in Y->U->V order (the layout `.ivf.md5` files use).
pub fn i420_bytes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.y.len() + frame.u.len() + frame.v.len());
    out.extend_from_slice(&frame.y);
    out.extend_from_slice(&frame.u);
    out.extend_from_slice(&frame.v);
    out
}
