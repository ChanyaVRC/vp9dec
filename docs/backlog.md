# Backlog

Open work items, ordered by priority. Each entry names its evidence/context so a fresh
session can pick it up cold. Completed milestones live in `history.md`; decisions and their
reasoning live in `implementation-notes.md`.

## P1 — SIMD optimization of the decode hot paths

Single-threaded scalar decode measures ~19 MP/s (1920-width content: 12-13 fps; 426-width:
245 fps). Realtime 1080p30 playback needs ~62 MP/s plus headroom — the target for HD content.

- Approach: `core::arch` intrinsics (AVX2 on x86_64, NEON on aarch64) with runtime
  detection (`is_x86_feature_detected!`) and the existing scalar code as the always-kept
  fallback — stays zero-dependency.
- Expected hot spots (to be CONFIRMED by profiling before writing any SIMD): subpel
  convolution (`predict.rs::block_inter_predict`), loop filter, inverse transforms,
  intra prediction. Wave 4b already replaced heap allocation with fixed scratch, so the
  data layout is SIMD-ready.
- Hard gate: bit-exact output — the full official sweep + the test suite + the ffmpeg
  cross-decode must stay green with SIMD enabled AND disabled (integer ops only, so
  exact equality is achievable; any "close enough" result is a bug).
- Wave 1 (measurement): DONE 2026-07-17 (`examples/bench.rs` + `bench-timing` feature).
  Baseline 35.7 MP/s; InterPredict 54.5% / LoopFilter 18.6% / Token+Dequant+Transform 17.7%.
- Wave 2 (AVX2 inter-pred subpel convolution): DONE 2026-07-17 (`src/simd.rs`).
  **35.7 -> 56.5 MP/s, 1.58x**, bit-exact (sweep passed both SIMD-on and forced-scalar).
  Scaled path + width-4 + edge blocks stay scalar.
- Wave 3 (AVX2 horizontal loop-filter edges): DONE 2026-07-18
  (`src/simd.rs::loop_filter_horiz8_avx2`), bit-exact both configs. After waves 2+3 the
  profile is balanced (LoopFilter / Token+Dequant+Transform / InterPredict each ~28-30%).
- Wave 3b (polish -- close the existing kernels' scalar fall-throughs): DONE 2026-07-21.
  The horizontal loop-filter kernel now handles the TX_16X16 "wide2" 16-tap case
  (`loop_filter_horiz8_avx2`, `is_tx16` lane mask + flat_mask2 + wide16 closed forms), and
  inter-pred now has a 128-bit width-4 kernel (`block_inter_predict_avx2_w4`) for the 4x4/4x8
  blocks that had stayed scalar. Bit-exact both SIMD configs (full sweep); unit-checked in
  `loop_filter.rs`/`simd.rs` (wide16 selection + w4-vs-w8 equivalence).
- Wave 4 (NEXT): the remaining scalar hot spots are the **vertical** loop-filter edges, the
  inverse transforms (`transform.rs` butterflies), and intra prediction. Same dispatch/bit-exact
  rules. Also still scalar: the 10/12-bit u16 path and the scaled (SVC/resize) inter path.
- NEON (aarch64) mirror: not started (x86_64 only so far); sibling module behind the same
  `predict.rs` dispatch point when an aarch64 target is needed.

## P2 — VP9 profiles 1-3 (4:2:2 / 4:4:4, 10/12-bit) — DONE 2026-07-19

Decoder support landed: the sweep is now 315/315 across all profiles, both SIMD-on and
forced-scalar. `Plane` is u16-backed; the public `Frame` exposes `enum PlaneData { U8, U16 }`
plus `bit_depth`/`subsampling_x/y`; the loop filter's 8-bit constants scale `<< (bit_depth-8)`
(identity at 8-bit, so 8-bit output stayed byte-identical). Profile 1 needed no code change
(the pipeline was already subsampling-general); the exploration's `residual.rs:158` "4:2:2 bug"
hypothesis was empirically refuted (applying it regressed the official 4:2:2 vector).

Remaining (moved to P1's SIMD scope, not blocking): a u16 SIMD path for 10/12-bit content.

## P3 — small recorded items

- **fetch script exit-code bug**: DONE 2026-07-22. `fetch-vectors.sh` captured `rc=$?` after
  the `fi`, reading the if-statement's status (always 0 when curl failed) instead of curl's, so
  the FAIL line always printed "curl exit 0"; moved the failure handling into an `else` branch
  where `$?` still holds curl's real status. (The PowerShell script reports via the exception
  and was already correct.)
- **ALTREF slot-steering test**: DONE 2026-07-22.
  `seg_lvl_ref_frame_steers_to_the_altref_slot` (`tests/synthetic_seg_test.rs`) plants distinct
  content in ALTREF's slot (physical slot 2 under `ref_frame_idx = [0, 1, 2]`) and forces
  `feature_data = ALTREF_FRAME`, asserting the decode copies that slot; LAST- and GOLDEN-steered
  companion probes pin that ALTREF is distinguished from *both* other single references. Cross-
  decoded byte-identically by ffmpeg's `libvpx-vp9` and native `vp9`. Completes the
  LAST/GOLDEN/ALTREF single-reference steering matrix (GOLDEN-vs-LAST was already proven).
- **Invalid-input robustness**: DONE 2026-07-21. A deterministic fuzz (`tests/robustness_test.rs`
  -- truncation + bit-corruption + random) guards against panics on malformed input, and the
  official libvpx `invalid-*` vector family (`kVP9InvalidFileTests`) is now a strict gate:
  `tests/invalid_vector_test.rs` decodes each vector frame-by-frame and requires an `Err` at
  exactly the packet libvpx's `.res` sidecar first rejects -- 21/21. (The family *is* hosted
  upstream; the earlier "not hosted" note was wrong -- fetched via the `invalid` manifest kind.)
  Reaching 21/21 added four validity checks the lenient decoder had lacked: a frame-size DoS
  guard (`HeaderError::FrameSizeTooLarge`), reference bit-depth/subsampling match
  (`RefFrameFormatMismatch`), and a per-tile over-read / under-read corruption gate
  (`TileError::CorruptTile`) -- all bit-exact-safe (full sweep unchanged). Still optional:
  structure-aware / longer fuzzing.

## P3 — retrospective "G group" residue (approved for the backlog 2026-07-17)

Items the 2026-07-16 design retrospective rated keep-as-documented; the standing
decision is now that they MAY be done. What "doing" each means, honestly:

- **`frame_context_idx` dual field**: DONE 2026-07-22. Folded to the single raw `f(2)` value
  plus a `NewFrameHeader::effective_frame_context_idx()` accessor (forces 0 when
  `FrameIsIntra || error_resilient_mode`); the `reset_frame_context == 2` `save_probs` path
  keeps using the raw value. Behaviour-preserving by construction (same u8 indices reach
  load/save/reset); a header unit test pins the accessor over all input combinations.
- **Out-of-line test files for big modules**: DONE 2026-07-22. Standardized on the split
  layout for the large modules (following `transform.rs`): `header`, `lib` (crate root ->
  `src/tests.rs`), `loop_filter`, `tile`, and `tile::mode_info` now keep their unit tests in a
  sibling `<module>/tests.rs` via `#[cfg(test)] mod tests;`. Convention recorded in
  `implementation-notes.md`. Smaller modules keep inline tests.
- **`prob_tables.rs` naming residue**: DECIDED keep (2026-07-22). A rename touches every
  `prob_tables::` import across the crate for a cosmetic gain; not worth the churn (the module
  holds probability/tree/geometry constant data, which the name adequately covers). Revisit only
  if the module is split.
- **examples/-as-tools & single-crate layout** (no action recommended): re-affirmed — still the
  right call at this size; revisit only if the tool count grows (then: `tools/` crate or
  workspace).

## Non-goals (decided, not deferred)

- Consulting libvpx / other decoder SOURCE for decode logic (clean-room rule; ffmpeg
  OUTPUT comparison as an oracle is fine and established).
