//! Full official-vector sweep (M4 wave 1): scans every `tests/vectors/*.ivf` that has a
//! matching `.ivf.md5` (fetched via `scripts/fetch-vectors.{sh,ps1}` from the manifest in
//! `scripts/vectors.txt`) and, for each, decodes every IVF chunk through one [`Decoder`],
//! MD5-checking each displayed frame's I420 bytes.
//!
//! Unlike `conformance_test.rs` (which asserts bit-exactness for 5 hand-curated vectors and
//! panics with a detailed diff on the first mismatch), this is a triage tool: one vector's
//! decode error, MD5 mismatch, or even a panic must not stop the sweep from covering the
//! rest, so every failure mode is caught and recorded rather than propagated. See
//! docs/implementation-notes.md "M4 wave 1" for the original failure categorization.
//!
//! The whole corpus now passes (315/315), so this is NOT `#[ignore]`d -- but it runs only in a
//! **release** build with the **full corpus present**, so it enforces conformance in CI /
//! `cargo test --release` without slowing routine debug `cargo test` (debug decode is ~10x
//! slower). A debug build or a partial checkout skips cleanly; the curated + profile-1-3
//! vectors are covered always-on in `conformance_test.rs` regardless. Enforcing run:
//! `RUST_MIN_STACK=16777216 cargo test --release --test sweep_test official_vector_sweep -- --nocapture`

mod common;

use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use common::md5::{md5, to_hex};
use vp9dec::ivf::IvfReader;
use vp9dec::Decoder;

/// Why one vector failed the sweep, as a short tag matching the mission's requested
/// categories (`md5-mismatch@frame N` / `error@frame N: E` / `panic` / `md5-count-mismatch`),
/// already rendered to a string (the panic payload and decode error need `Debug`/message
/// extraction that's easiest to do once, at the point each is caught).
enum Failure {
    Mismatch(String),
}

/// Appends `.md5` to an `.ivf` path's full file name (not [`Path::with_extension`], which
/// would only be correct here because these paths happen to end in exactly `.ivf` --
/// appending is correct regardless of how many dots are in the vector's own name, e.g.
/// `vp90-2-00-quantizer-00.webm.ivf.s5861_r01-05_b6-.v2.ivf`).
fn md5_sidecar_path(ivf_path: &Path) -> PathBuf {
    let mut name = ivf_path.as_os_str().to_owned();
    name.push(".md5");
    PathBuf::from(name)
}

/// Decodes every IVF chunk in `ivf_path` through one [`Decoder`], MD5-checking each displayed
/// frame's I420 bytes against the corresponding line of `md5_path`. Returns `Ok(())` iff every
/// displayed frame matched and the output frame count equals the `.ivf.md5` line count;
/// otherwise `Err` describing the first problem found. Never panics itself on decode failure
/// (decode errors are returned, not unwrapped) -- a genuine panic (e.g. an internal `unwrap`
/// or index out of bounds inside the decoder) still unwinds out of this function, and is
/// caught by the `catch_unwind` around this function's call site instead.
fn sweep_one(ivf_path: &Path, md5_path: &Path) -> Result<(), Failure> {
    let ivf_bytes =
        std::fs::read(ivf_path).map_err(|e| Failure::Mismatch(format!("read-error: {e}")))?;
    let md5_text = std::fs::read_to_string(md5_path)
        .map_err(|e| Failure::Mismatch(format!("md5-read-error: {e}")))?;
    let expected: Vec<String> = md5_text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_ascii_lowercase)
        .collect();

    let reader = IvfReader::new(&ivf_bytes)
        .map_err(|e| Failure::Mismatch(format!("ivf-header-error: {e:?}")))?;

    let mut decoder = Decoder::new();
    let mut output_idx = 0usize;

    for (ivf_frame_idx, frame) in reader.enumerate() {
        let frame = frame
            .map_err(|e| Failure::Mismatch(format!("ivf-frame-error@{ivf_frame_idx}: {e:?}")))?;
        let constituent_frames = decoder
            .decode_frame(frame.data)
            .map_err(|e| Failure::Mismatch(format!("error@frame {ivf_frame_idx}: {e:?}")))?;
        for df in constituent_frames {
            let Some(decoded) = df.frame else { continue };
            let actual = to_hex(&md5(&common::i420_bytes(&decoded)));
            let Some(expected_hash) = expected.get(output_idx) else {
                return Err(Failure::Mismatch(format!(
                    "md5-count-mismatch: produced more than {} output frame(s)",
                    expected.len()
                )));
            };
            if &actual != expected_hash {
                return Err(Failure::Mismatch(format!(
                    "md5-mismatch@frame {output_idx}"
                )));
            }
            output_idx += 1;
        }
    }

    if output_idx != expected.len() {
        return Err(Failure::Mismatch(format!(
            "md5-count-mismatch: produced {output_idx} output frame(s), expected {}",
            expected.len()
        )));
    }
    Ok(())
}

/// Number of md5-checkable vectors in the full official corpus (profile 0-3), as of 2026-07.
/// The sweep runs only when at least this many are present, so a partial checkout skips rather
/// than passing a small subset and looking like full conformance. Bump if the corpus grows.
const FULL_CORPUS_MD5_MIN: usize = 300;

#[test]
fn official_vector_sweep() {
    // Release-only: debug decode is ~10x slower, so sweeping the whole corpus in a debug
    // `cargo test` would add many minutes to every routine run. In debug we skip (the curated +
    // profile-1-3 vectors still run always-on in conformance_test.rs); the enforcing run is
    // `cargo test --release`, which CI/conformance uses and which is where this gate lives now
    // that it's no longer #[ignore]d.
    if cfg!(debug_assertions) {
        eprintln!(
            "[skip] debug build -- run the full-corpus sweep with `cargo test --release` \
             (debug decode is ~10x slower; the curated + profile-1-3 vectors are covered \
             always-on in conformance_test.rs)"
        );
        return;
    }

    let vectors_dir = common::vectors_dir();
    let mut ivf_paths: Vec<PathBuf> = std::fs::read_dir(&vectors_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", vectors_dir.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "ivf"))
        .filter(|p| md5_sidecar_path(p).exists())
        .collect();
    ivf_paths.sort();

    if ivf_paths.len() < FULL_CORPUS_MD5_MIN {
        eprintln!(
            "[skip] only {} md5-checkable *.ivf present under {} (< {FULL_CORPUS_MD5_MIN}); \
             the full-corpus sweep needs `scripts/fetch-vectors.{{sh,ps1}}` first \
             (the curated + profile-1-3 vectors are covered always-on in conformance_test.rs)",
            ivf_paths.len(),
            vectors_dir.display()
        );
        return;
    }

    // Silenced for the duration of the sweep: a decoder-internal panic is an expected outcome
    // here (its own failure category), not a test-harness bug, and printing the default
    // backtrace-style hook output once per panicking vector would drown out the report below.
    // catch_unwind still receives the payload regardless of the hook.
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut report_lines: Vec<String> = Vec::with_capacity(ivf_paths.len());
    let mut pass = 0usize;
    let mut fail_md5_mismatch = 0usize;
    let mut fail_count_mismatch = 0usize;
    let mut fail_decode_error = 0usize;
    let mut fail_panic = 0usize;

    for ivf_path in &ivf_paths {
        let name = ivf_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| ivf_path.display().to_string());
        let md5_path = md5_sidecar_path(ivf_path);

        let result = panic::catch_unwind(AssertUnwindSafe(|| sweep_one(ivf_path, &md5_path)));

        let line = match result {
            Ok(Ok(())) => {
                pass += 1;
                format!("[PASS] {name}")
            }
            Ok(Err(Failure::Mismatch(reason))) => {
                if reason.starts_with("md5-count-mismatch") {
                    fail_count_mismatch += 1;
                } else if reason.starts_with("md5-mismatch") {
                    fail_md5_mismatch += 1;
                } else {
                    fail_decode_error += 1;
                }
                format!("[FAIL <{reason}>] {name}")
            }
            Err(payload) => {
                fail_panic += 1;
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                format!("[FAIL <panic: {msg}>] {name}")
            }
        };
        eprintln!("{line}");
        report_lines.push(line);
    }

    panic::set_hook(default_hook);

    let total = ivf_paths.len();
    let fail = total - pass;
    let summary = format!(
        "\n===== official_vector_sweep summary =====\n\
         total:              {total}\n\
         pass:               {pass}\n\
         fail:               {fail}\n\
         fail md5-mismatch:  {fail_md5_mismatch}\n\
         fail count-mismatch:{fail_count_mismatch}\n\
         fail decode-error:  {fail_decode_error}\n\
         fail panic:         {fail_panic}\n\
         ===========================================\n"
    );
    eprintln!("{summary}");
    report_lines.push(summary);

    let report_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("sweep-report.txt");
    if let Some(parent) = report_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&report_path, report_lines.join("\n"))
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", report_path.display()));
    eprintln!("[report] written to {}", report_path.display());

    assert_eq!(
        pass,
        total,
        "{fail}/{total} vector(s) failed the sweep -- see {} for the full per-vector report",
        report_path.display()
    );
}
