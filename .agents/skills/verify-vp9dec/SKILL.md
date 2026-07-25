---
name: verify-vp9dec
description: Acceptance checklist to run before finalizing any change to this VP9 decoder. Covers the bit-exact gate (the full official conformance sweep in both SIMD configurations, plus independent ffmpeg cross-decode), how to tell a real pass from a vacuous [skip], and the rule to trust the empirical conformance sweep over static reasoning.
---

# vp9dec acceptance checklist

A from-scratch, zero-dependency VP9 decoder whose entire value is bit-exact conformance.
Verify a change by re-running the gates and reading the evidence — not by trusting a summary
or a green `test result: ok` line.

## Ground rules

- **Never assert a fixed total test count.** It drifts as tests are added/removed/merged.
  Record the actual number each run; if it shifts unexpectedly, diff `cargo test -- --list`
  to name the functions that appeared or vanished. "0 failed but the count dropped" is not OK
  until explained.
- **`src/` stays zero external crates.** Confirm `git diff Cargo.toml` is empty (no dependency
  added). Test-only dev-dependencies are allowed but rarely needed.

## 1. Default suite

- `cargo test` all green. `conformance_test` runs by default and MD5-checks the curated
  profile-0 vectors plus the profile 1/2/3 vectors — if it is green, basic profile 0-3
  conformance is guarded.
- **Spot the vacuous `[skip]`.** The conformance test, the full sweep, and the ffmpeg
  cross-decode all *skip cleanly and pass* when their vectors / ffmpeg are absent, so a green
  run can prove nothing. Don't stop at `test result: ok`; grep for the evidence that they
  actually ran (`[ok]` / `[xdecode]` lines, real pass counts). If conformance prints `[skip]`,
  the vectors aren't fetched — run `scripts/fetch-vectors.{sh,ps1}`.

## 2. Bit-exact gate (required for any decode-path / SIMD / refactor change)

**8-bit output must stay byte-identical.** A pixel-format widening, ownership change, SIMD
kernel, or module move must not alter a single output byte of any 8-bit decode.

- Run the full official sweep in **release** — it passes 315/315:
  `cargo test --release --test sweep_test official_vector_sweep -- --nocapture` →
  `total: 315 / pass: 315 / fail: 0`. (It is release-only and full-corpus-only by design; a
  debug build or a partial checkout skips it — debug decode is ~10x slower.)
- **For SIMD / decode-path changes, run it twice**: once normally, once with
  `VP9DEC_NO_SIMD=1`. Both at 315/315 proves *SIMD output == scalar output == official MD5*.
- A faster-but-different result from a SIMD/optimization path is a **bug, not a tradeoff** —
  one differing byte fails acceptance.

## 3. Independent ffmpeg cross-decode

- `cargo test --test synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`
  → one `[xdecode] ... OK` line per synthetic scenario × decoder (`libvpx-vp9` and native
  `vp9` — so 2 lines per scenario, every one `OK`). Don't assert a fixed total (the scenario
  count grows); every emitted line must be `OK` and there must be at least one.
- Needs `ffmpeg` on `PATH`, or an explicit `VP9DEC_FFMPEG=<path-to-ffmpeg>`; it skips cleanly
  if neither is present. **0 lines means it didn't run**, not that it passed.

## 4. Lint / format / docs

- `cargo clippy --all-targets`: no **new** warnings beyond the repo's small known baseline.
- `cargo fmt --check` clean.
- `docs/implementation-notes.md` is updated only if the change adds a still-relevant landmine,
  non-obvious rationale, or known gap — keep it concise (resolved-bug detail lives in git
  history, not the notes). Deferred work goes in `docs/backlog.md`.

## Principle: empirical bit-exactness beats static reasoning

Do **not** "fix" conformance-passing code because a spec reading or another decoder's source
says it "should" be different — verify with the sweep first. A plausible, source-grounded
change to the sub-8x8 chroma MV index (meant to fix 4:2:2) once *regressed* the official 4:2:2
vector: the original was already bit-exact. When ground truth (the sweep) and a hypothesis
disagree, the sweep wins.

## Reject criteria

Send back / fix if any gate is unmet or a report disagrees with a re-run — in particular:
the 8-bit sweep is not 315/315 in **both** SIMD configurations; a new-profile vector
mismatches; any ffmpeg cross-decode line is missing or not `OK`; or a "pass" was really a
silent `[skip]` that never ran.
