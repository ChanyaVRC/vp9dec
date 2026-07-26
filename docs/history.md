# History

How the decoder was built, milestone by milestone — a narrative record of the clean-room
implementation order, not current documentation. For the decoder as it stands now, see
[README.md](../README.md) (architecture, API, status) and
[implementation-notes.md](implementation-notes.md) (current rationale, landmines, gaps). Detail
finer than the summaries below lives in the git log.

## Milestones

| Milestone | What landed |
| --- | --- |
| M1 | IVF container parser, bool (arithmetic) decoder, uncompressed frame-header parsing. |
| M2 | Key-frame intra decode: compressed header, tiles, token decoding, inverse quant/transform, reconstruction, all 10 VP9 intra modes. |
| M2b | Deblocking loop filter (spec §8.8) and the first official MD5 conformance check (against a from-scratch MD5). |
| M3 first half | Inter-frame bitstream decode: non-key header, inter probability tables, mode info, MV prediction, residual tokens — everything up to but not including pixel generation. |
| M3 second half | Motion compensation (subpel interpolation, compound, reference scaling), the 8-slot DPB, forward/backward probability adaptation, loop-filter deltas — full multi-frame MD5 conformance. |
| M4 | The full official-vector sweep, then profiles 1-3 (10/12-bit and 4:2:2 / 4:4:0 / 4:4:4). 315/315 vectors bit-exact. |
| M5 | Decode performance: all-depth AVX2 for inter prediction, loop filtering, and every non-lossless inverse transform; tile-column parallel decode and a loop-filter wavefront. The sweep remained 315/315 with SIMD enabled and forced scalar; 1080p reached about 98 MP/s single-tile and 155 MP/s with four tiles. |
| M6 | Hardening and maintenance: official invalid vectors reached 21/21, deterministic malformed-input fuzzing guarded against panics, decoder structure was consolidated, and unit-test bodies moved under `tests/unit/`. |
| M7 | Coverage close-out: a wide profile-3 HBD stream covered multi-tile `U16` strips; structure-aware fuzzing reached stateful entropy paths; 27 generated conformant scenarios compared SIMD with forced scalar in isolated processes; and the last material x86 inter-prediction fallback (unscaled reference edges) moved to AVX2. Measurement kept intra prediction and lossless WHT scalar, while NEON remains conditional on a concrete aarch64 target. |

## Notes from the build

Things worth remembering about *how* it was done (the design decisions that still shape the
code are in [implementation-notes.md](implementation-notes.md); the current architecture is in
[README.md](../README.md)):

- **Clean-room, from the spec only.** Decode logic came from the VP9 Bitstream & Decoding
  Process Specification v0.7 — no other decoder's source. Large numeric tables
  (`DEFAULT_COEF_PROBS`, `PARETO_TABLE`, `SUBPEL_FILTERS`, …) were extracted mechanically from
  the spec PDF (`pdftotext` + a regex) rather than hand-transcribed, to avoid transcription
  errors.
- **MD5 conformance was the spine from M2b on.** Every milestone was gated on the official
  `.ivf.md5` matching the decoder's I420 output byte-for-byte; the from-scratch MD5
  (`tests/common/md5.rs`) exists for exactly that.
- **Two local vectors carried M2b–M3; the full corpus arrived in M4.** `vp90-2-12-droppable_1`
  and `vp90-2-09-subpixel-00` passed every displayed frame frame-exact through M3, then M4
  expanded to the whole official set (now fetched by `scripts/fetch-vectors.{sh,ps1}`).

The public API, module layout, and test files were reshaped repeatedly after M3 (the
design-debt waves and the profile work), so any struct / function / file named above may since
have moved or changed. README.md and implementation-notes.md are authoritative for the current
shape.
