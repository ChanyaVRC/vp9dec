# vp9dec

A from-scratch, clean-room VP9 video decoder in Rust. Its entire value is **bit-exact
conformance**: every official test vector must decode to the exact bytes libvpx produces.

## Load-bearing invariants (do not break these)

- **Clean-room.** Decode logic is derived from the [VP9 Bitstream & Decoding Process
  Specification v0.7](https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf)
  only. Do **not** consult libvpx (or any other decoder's) source for decode logic.
- **`src/` uses zero external crates** — Rust std only, no runtime dependencies. Test tooling
  may use `[dev-dependencies]`, but in practice the only one is a self-referential dev-dep on
  this crate (`features = ["test-support"]`). Adding a real external crate to `src/` is out of
  scope.
- **8-bit output is bit-exact and must stay so.** A refactor, ownership change, SIMD kernel, or
  any "optimization" must not alter a single output byte of any 8-bit decode. The official
  sweep is the gate, in both SIMD configurations.
- **Trust the empirical sweep over static / spec reasoning.** Do not "fix" conformance-passing
  code because a spec reading or another decoder says it "should" differ — verify against the
  sweep first. When a hypothesis and the sweep disagree, the sweep wins.
- **Keep `docs/implementation-notes.md` concise** — it holds still-relevant design rationale,
  landmines, and known gaps, not a blow-by-blow log. A resolved bug needs no entry; that detail
  lives in git history.

## Build & test

```sh
cargo test                 # default suite; conformance/sweep skip cleanly if vectors absent
cargo clippy --all-targets
cargo fmt --check
```

Conformance vectors aren't committed — fetch them first, then run the full sweep (release):

```sh
scripts/fetch-vectors.sh   # or scripts/fetch-vectors.ps1 (idempotent, ~3.5 GB)
cargo test --release --test sweep_test official_vector_sweep -- --nocapture   # -> 315/315
```

Before finalizing any decode-path change, run the full acceptance gate — see the
**`verify-vp9dec`** skill.

## Where to look

- **Architecture & module map** → [README.md](README.md) ("Current architecture").
- **Public API, core output types, change-navigation, and gotchas** → the
  **`vp9dec-architecture`** skill.
- **Design rationale, landmines, and known gaps** →
  [docs/implementation-notes.md](docs/implementation-notes.md).
- **Deferred work** → [docs/backlog.md](docs/backlog.md).
