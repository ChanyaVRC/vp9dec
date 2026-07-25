# Backlog

Open work only, ordered by priority. Completed milestones live in
[history.md](history.md); current design rationale, landmines, and known gaps live in
[implementation-notes.md](implementation-notes.md). Finer detail belongs in git history.

## P2 — Robustness and portability

- **Structure-aware / long-running fuzzing.** Extend the deterministic truncation,
  bit-corruption, and random-input coverage with mutations that preserve enough VP9 structure
  to reach deeper decode states, plus an opt-in long-running mode. Keep the official invalid
  vector gate at 21/21 and require no panics.
- **aarch64 NEON.** Mirror the currently profiled AVX2 families only when an aarch64 target is
  needed. Keep runtime dispatch, the scalar fallback, zero dependencies, and bit-exact output.

## P3 — Measurement-gated follow-ups

- **Remaining scalar decode paths.** Intra prediction, the lossless 4x4 WHT, and unscaled
  inter prediction near reference edges remain scalar. Profile the intended workload before
  optimizing: intra prediction measured about 0.3% on inter content, and WHT is a narrow
  lossless-only path.
- **Conformant SIMD/scalar differential testing.** A generated-stream differential mode could
  explore more valid combinations than the fixed corpus. Restrict comparisons to conformant
  streams: malformed coefficients may legitimately wrap differently in SIMD and scalar
  arithmetic, so arbitrary-corruption output equality is not a valid oracle.
