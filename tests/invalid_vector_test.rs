//! Strict gate over libvpx's official invalid-input vectors (`kVP9InvalidFileTests`), fetched
//! by `scripts/fetch-vectors.{sh,ps1}` (kind `invalid`) into `tests/vectors/` as
//! `invalid-*.ivf` alongside a `<ivf-name>.res` sidecar.
//!
//! A `.res` file lists libvpx's expected per-packet decode result code, one per line
//! (`vpx_codec_err_t`: 0 = OK, non-zero = a rejection -- e.g. 2 MEM_ERROR, 5 UNSUP_BITSTREAM,
//! 7 CORRUPT_FRAME). This gate holds our decoder to the leading edge of that list: for the
//! frames libvpx decoded cleanly (the `0` prefix) our [`Decoder::decode_frame`] must return
//! `Ok`, and at the first frame libvpx rejected (the first non-zero code) ours must return
//! `Err` -- then we stop. (libvpx continues past that frame with its own frame-parallel error
//! recovery, sometimes returning `0` again; reproducing that recovery is out of scope -- the
//! gate is the reject-at-the-right-frame contract.)
//!
//! Unlike the fuzz in `robustness_test.rs` (which only demands "don't panic"), this demands the
//! decoder actually *detect* each corruption at the frame libvpx does. It runs by default (the
//! vectors are tiny and error early) and skips cleanly when they aren't fetched.

mod common;

use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use vp9dec::ivf::IvfReader;
use vp9dec::Decoder;

/// One expected result code per decoded packet, parsed from a `.res` sidecar (one integer per
/// non-blank line; the first whitespace token of each line).
fn parse_res(path: &Path) -> Vec<i32> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    text.lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(|tok| {
            tok.parse::<i32>()
                .unwrap_or_else(|_| panic!("{}: non-integer .res entry {tok:?}", path.display()))
        })
        .collect()
}

/// Applies the strict gate to one vector. `Ok(())` = it met the contract; `Err(reason)` names
/// the first way it didn't. Each `decode_frame` runs inside `catch_unwind` so a decoder panic
/// on one packet is reported (as `panic@i`), never aborts the whole test.
fn gate_one(ivf_bytes: &[u8], res: &[i32]) -> Result<(), String> {
    // Index of the first packet libvpx rejected; every earlier packet it decoded cleanly.
    let first_reject = res.iter().position(|&c| c != 0);

    let reader = IvfReader::new(ivf_bytes).map_err(|e| format!("ivf-header-error: {e:?}"))?;
    let mut dec = Decoder::new();

    for (i, frame) in reader.enumerate() {
        // Past the first rejection we stop; before it, this frame must decode; at it, must fail.
        let at_reject = first_reject == Some(i);
        if let Some(k) = first_reject {
            if i > k {
                break;
            }
        }

        let frame = match frame {
            Ok(f) => f,
            Err(e) => {
                // The container itself couldn't yield this packet. Acceptable iff this is where
                // libvpx also rejected the stream (rejection at the container layer counts).
                if at_reject {
                    return Ok(());
                }
                return Err(format!("ivf-frame-error@{i} (expected decodable): {e:?}"));
            }
        };

        let decoded = panic::catch_unwind(AssertUnwindSafe(|| dec.decode_frame(frame.data)));
        match decoded {
            Err(_) => return Err(format!("panic@{i}")),
            Ok(Ok(_)) => {
                if at_reject {
                    return Err(format!(
                        "decoded OK@{i}, but libvpx rejects here (res code {})",
                        res[i]
                    ));
                }
                // Expected OK, decoded OK -- continue to the next packet.
            }
            Ok(Err(e)) => {
                if at_reject {
                    return Ok(()); // rejected exactly where libvpx does
                }
                return Err(format!(
                    "rejected@{i} ({e:?}), but libvpx decodes here (res code 0)"
                ));
            }
        }
    }

    match first_reject {
        Some(k) => Err(format!(
            "stream ended before reaching the expected rejection at frame {k}"
        )),
        None => Ok(()), // an all-OK .res (no vector actually ships one) -- nothing to reject
    }
}

#[test]
fn official_invalid_vector_gate() {
    let vectors_dir = common::vectors_dir();
    let mut vectors: Vec<PathBuf> = match std::fs::read_dir(&vectors_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("invalid-") && n.ends_with(".ivf"))
            })
            .filter(|p| common::sidecar_path(p, "res").exists())
            .collect(),
        Err(_) => Vec::new(),
    };
    vectors.sort();

    if vectors.is_empty() {
        eprintln!(
            "[skip] no invalid-*.ivf with a .res sidecar under {} -- run \
             scripts/fetch-vectors.{{sh,ps1}} (the `invalid` manifest entries)",
            vectors_dir.display()
        );
        return;
    }

    // A caught decode panic is an expected outcome to report here, not a harness bug; silence the
    // default backtrace-printing hook for the run (catch_unwind still receives the payload).
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut pass = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for ivf in &vectors {
        let name = ivf.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(ivf).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let res = parse_res(&common::sidecar_path(ivf, "res"));
        match gate_one(&bytes, &res) {
            Ok(()) => {
                pass += 1;
                eprintln!("[ok] {name} res={res:?}");
            }
            Err(reason) => {
                eprintln!("[FAIL] {name}: {reason} (res={res:?})");
                failures.push(format!("{name}: {reason}"));
            }
        }
    }

    panic::set_hook(default_hook);

    eprintln!(
        "\n===== official_invalid_vector_gate: {pass}/{} passed =====",
        vectors.len()
    );
    assert!(
        failures.is_empty(),
        "{} invalid vector(s) failed the strict gate:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
