# Backlog

Open work items, ordered by priority. Each entry names its evidence/context so a fresh
session can pick it up cold. Completed milestones live in `history.md`; decisions and
their reasoning live in `implementation-notes.md` (see its Current state index).

## P1 — SIMD optimization of the decode hot paths

Single-threaded scalar decode measures ~19 MP/s (1920-width content: 12-13 fps; 426-width:
245 fps — see implementation-notes "M4 final wave" for the measurement). Realtime 1080p30
needs ~62 MP/s plus headroom, so this blocks practical Noiria integration for HD content.

- Approach: `core::arch` intrinsics (AVX2 on x86_64, NEON on aarch64) with runtime
  detection (`is_x86_feature_detected!`) and the existing scalar code as the always-kept
  fallback — stays zero-dependency.
- Expected hot spots (to be CONFIRMED by profiling before writing any SIMD): subpel
  convolution (`predict.rs::block_inter_predict`), loop filter, inverse transforms,
  intra prediction. Wave 4b already replaced heap allocation with fixed scratch, so the
  data layout is SIMD-ready.
- Hard gate: bit-exact output — 304/304 official sweep + 150-test suite + 8-way ffmpeg
  cross-decode must stay green with SIMD enabled AND disabled (integer ops only, so
  exact equality is achievable; any "close enough" result is a bug).
- First wave = measurement: DONE (2026-07-17, `examples/bench.rs` +
  `bench-timing` feature; see implementation-notes "SIMD wave 1"). Result (release,
  inter-heavy 1920x800 movie): **35.7 MP/s single-thread**, split InterPredict 54.5% /
  LoopFilter 18.6% / Token+Dequant+Transform 17.7% / Intra 0.4%. Wave-2 target order is
  therefore: (1) `predict.rs::block_inter_predict` 8-tap subpel convolution [dominant],
  (2) `loop_filter.rs`, (3) `transform.rs` butterflies. `is_x86_feature_detected!`
  dispatch, scalar fallback kept, bit-exact gate (integer ops).

## P1.5 — Noiria integration: in-engine validation

CORRECTION (2026-07-17): the integration itself already exists — Noiria's
`crates/noiria-core/src/codec/video.rs` has a `Vp9Ivf` `VideoSource` backed by this
crate (path dependency), sniffed by `open_video_file()` ahead of Media Foundation. The
2026-07-16 API break was adapted there on 2026-07-17 (Noiria workspace fully green).
Remaining work: in-engine playback validation of HD content once P1 (SIMD) lands, and
whatever the Noiria-side backlog's frame-seek discussion (its B4) decides. When changing
this crate's API, run Noiria's `cargo check -p noiria-core` as the break detector.
8-bit 4:2:0 only until P2 lands. HEVC stays Media Foundation (patents — separate
decision, not backlog).

## P2 — VP9 profiles 1-3 (4:2:2 / 4:4:4, 10/12-bit)

Approved for the backlog 2026-07-17 (previously out of scope). No patent concern; pure
implementation volume. Large:

- `Plane` is `u8`-fixed — 10/12-bit needs u16 pixel storage through framebuffer,
  predict, transform ranges, loop filter, MC, DPB, and output (`Frame` API shape
  change for >8-bit output — design decision needed).
- 4:2:2/4:4:4 relaxes the subsampling assumptions wherever chroma is halved today
  (`subsampling_x/y` are already parsed and threaded; the hardcoded assumptions are in
  plane sizing and MC/loop-filter chroma paths).
- Gates exist upstream: the official `vp91-2-*` / `vp92-2-*` / `vp93-2-*` vector
  families (extend `scripts/vectors.txt` + the sweep).
- Suggested order: profile 2 (10/12-bit 4:2:0) first — touches depth only; then
  profiles 1/3 (subsampling) on top.

## P3 — small recorded items

- **bbb clips ffmpeg cross-check** (small): the 7 `vp90-2-bbb_*` clips have no upstream
  `.md5` and were NOT cross-checked against ffmpeg (the 12 tos/sintel were —
  268,832 frames byte-identical). Run the same framemd5 comparison once.
- **fetch script exit-code bug** (trivial): `rc=$?` is captured after the `fi`, reading
  the if-statement's status instead of curl's (recorded in implementation-notes
  "M4 final wave"; cosmetic — failures still surface via the summary).
- **ALTREF slot-steering test** (small): SEG_LVL_REF_FRAME steering is proven for
  GOLDEN-vs-LAST; an ALTREF-direction case would complete the matrix
  (implementation-notes "Synthetic round-trip coverage", noted as future addition).
- **Invalid-input robustness** (medium, conditional): the decoder trusts conformant
  input by design (documented judgment calls). If Noiria ever plays user-supplied
  files rather than shipped assets, run the libvpx `invalid-*` family (generated
  locally, not hosted) and/or fuzz the container/header layer.

## P3 — retrospective "G group" residue (approved for the backlog 2026-07-17)

Items the 2026-07-16 design retrospective rated keep-as-documented; the standing
decision is now that they MAY be done. What "doing" each means, honestly:

- **`frame_context_idx` dual field** (small): store only the raw `f(2)` value + an
  `effective_idx()` accessor (one source of truth). The dual field was a deliberate
  blast-radius call during a 2026-07-12 bugfix; folding it is safe now that
  conformance pins behavior.
- **Out-of-line test files for big modules** (small, cosmetic): `transform.rs` +
  `transform/tests.rs` is the only module with the (nicer) split layout; standardize
  the convention for the other large modules, or explicitly re-affirm inline tests.
- **`prob_tables.rs` naming residue** (trivial): W5 already extracted subpel/MV tables;
  what remains really is probability/tree/geometry data. Optional rename or leave.
- **examples/-as-tools & single-crate layout** (no action recommended): re-evaluated in
  the retrospective and still the right call at this size; revisit only if the tool
  count grows (then: `tools/` crate or workspace).

## Non-goals (decided, not deferred)

- HEVC decoding (patents — stays Media Foundation on the Noiria side).
- Consulting libvpx/other decoder SOURCE for decode logic (clean-room rule; ffmpeg
  OUTPUT comparison as an oracle is fine and established).
