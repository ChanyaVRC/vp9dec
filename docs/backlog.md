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
- Wave 4a (AVX2 vertical loop-filter edges): DONE 2026-07-22
  (`src/simd.rs::loop_filter_vert8_avx2` + `superblock_loop_filter_vert_edge_avx2`). Transposes
  the tap window (8x8 / 8x16) into the row-major layout the horizontal kernel wants, reuses that
  proven kernel unchanged, and transposes back -- so all mask / narrow / wide8 / wide16
  arithmetic is shared verbatim; only the load/store orientation differs. Bit-exact both SIMD
  configs (full sweep 315/315) + ffmpeg cross-decode; a 500-trial `loop_filter/tests.rs` unit
  test pins the kernel against the scalar `sample_filtering`. **37.65 -> 41.63 MP/s, 1.11x** on
  1920x800 (tos, 17620 frames). Both loop-filter passes are now AVX2.
- Wave 4b (AVX2 8-bit DCT_DCT inverse transform, all sizes 4/8/16/32): DONE 2026-07-23
  (`src/simd.rs::inverse_transform_dct_dct_reconstruct_avx2`, `idct_pass` / `idct_simd` /
  `transpose*_i32`; see the Wave 4b follow-up below for the fused reconstruction).
  The scalar recursive `transform::idct` is mirrored verbatim on i32 8-lane vectors (8 rows in
  parallel); the 2D driver runs two transpose-on-store passes (row -> R^T, column -> O with
  round2). i32 lane storage is bit-exact for 8-bit because spec §8.7.1.1 bounds intermediates to
  `8 + BitDepth == 16` bits, so every `t*cos64` product fits i32. Dispatched from
  `tile/residual.rs` only for `bit_depth == 8 && tx_type == DctDct && !lossless`; ADST / WHT /
  mixed and 10/12-bit stay scalar. Bit-exact both SIMD configs (full sweep 315/315) + ffmpeg
  cross-decode; unit tests pin the 1D idct and the full 2D vs scalar for every size.
  **45.07 -> 54.46 MP/s, 1.21x** on 854x356 (the transform was 23.8% of decode -- a profiled
  `InverseTransform` sub-timer; ~3.6x on the transform itself).
- Wave 4b follow-up (fuse reconstruction): DONE 2026-07-23. The DCT_DCT path now runs
  `inverse_transform_dct_dct_reconstruct_avx2`, adding the column-pass residual straight into the
  plane with the 8-bit clip -- dropping both the i64 write-back and the scalar per-pixel
  reconstruction loop. Bit-exact both configs (sweep 315/315); +2.2% on 854x356. A re-profile put
  the remaining scalar hot spots at token/entropy decode (~sequential, not SIMD-able) and
  reconstruction (now fused); the big-stage InterPredict and LoopFilter are already AVX2.
- 10/12-bit inter-prediction SIMD: DONE 2026-07-23. The AVX2 inter-pred kernels
  (`block_inter_predict_avx2` / `_w4`) already work in i32 (which holds a 12-bit sample through
  both FIR passes); the only 8-bit-specific value was the `clip1` bound, now passed as
  `max_val = (1<<bit_depth)-1`, and the `predict.rs` dispatch no longer gates on `bit_depth == 8`.
  Bit-exact (sweep 315/315 both configs, incl. the `vp9{2,3}-2-20-1{0,2}bit-*` vectors). No large
  high-bit-depth clip exists to bench precisely (the conformance HBD clips are ~160x90x10 frames).
- 10/12-bit loop-filter SIMD: DONE 2026-07-23. The AVX2 loop-filter kernels
  (`loop_filter_horiz8_avx2` / `_vert8_avx2`) take `bit_depth` and scale their three 8-bit
  constants -- narrow base `1<<(bit_depth-1)`, clamp range `+/-(128<<(bit_depth-8))`, flat
  threshold `1<<(bit_depth-8)` -- matching the scalar `narrow_filter` / `compute_filter_mask`
  (limit/blimit/thresh were already bit-depth values from `adaptive_filter_strength`). The
  `superblock_loop_filter` dispatch no longer gates on `bit_depth == 8`. Bit-exact (sweep 315/315
  both configs, incl. the 10/12-bit vectors).
- 10/12-bit DCT_DCT inverse-transform SIMD: DONE 2026-07-24. The idct network keeps its i32
  lane storage at every depth (spec §8.7.1.1 bounds all stored intermediates to signed
  `8 + BitDepth <= 20` bits), so only the butterfly's `t*cos64` products (~2^33 at 12-bit)
  needed widening: `simd.rs::b_op_simd_hbd` computes them with 32x32->64-bit `_mm256_mul_epi32`
  multiplies and rounds in i64; the rest of the network is shared verbatim via
  `idct_simd::<const HBD>` (monomorphized, so the 8-bit path's codegen is unchanged). The new
  fused entry `inverse_transform_dct_dct_reconstruct_hbd_avx2` clips at `(1<<bit_depth)-1`;
  the `tile/residual.rs` dispatch routes 10/12-bit DCT_DCT (non-lossless) there, 8-bit to the
  existing kernel; ADST / WHT / mixed and lossless stay scalar. Bit-exact (sweep 315/315 both
  SIMD configs, incl. the `vp92-2-20-10bit-*` / `vp93-2-20-12bit-*` vectors); unit tests pin
  the 1D HBD idct (at full ±2^19 12-bit magnitudes, with a self-check that those magnitudes
  overflow the i32-product network) and the fused 2D at 10-bit and 12-bit, every size, against
  the scalar. Perf: no large high-bit-depth clip exists (the conformance HBD clips are
  ~160x90x10 frames, ~8 ms/decode), so whole-clip MP/s vs master is within noise; the
  `InverseTransform` sub-timer on the 12-bit clip went 1.0 ms -> 0.7 ms (9.8% -> 7.2% of
  decode) -- directional only. The 8-bit path is untouched (separate monomorphization;
  854x356 clip re-benched at par).
- 8-bit ADST inverse-transform SIMD (ADST_DCT / DCT_ADST / ADST_ADST, sizes 4/8/16 -- 32x32 is
  DCT-only): DONE 2026-07-24 (`simd.rs::inverse_transform_adst_reconstruct_avx2` +
  `iadst4/8/16_simd` / `sb_op_simd` / `sh_op_simd`; the separable pass is now
  `xform_pass::<HBD, ADST>` and the fused add+clip reconstruction is shared via
  `reconstruct_add_clip`). The scalar `transform::iadst*` networks are mirrored verbatim on i32
  8-lane vectors. 8-bit ONLY: the spec §8.7.1.1/§8.7.2 bounds (|T| <= 2^15, |cos64| <= 2^14)
  keep every SB product (<= 2^29), unrounded S value (< 2^30), SH sum (< 2^31) and iadst4 chain
  (43801 * 2^15 < 2^31) inside i32, but at 10/12-bit the S array needs `24 + BitDepth` (up to
  36) bits of LANE STORAGE, so the DCT's products-only i64 widening does not carry over --
  10/12-bit ADST stays scalar, as does lossless WHT (tiny 4x4-only lossless path whose
  shift/no-round structure shares nothing with the butterfly infra). Bit-exact (sweep 315/315
  both SIMD configs) + ffmpeg cross-decode 10/10; unit tests pin the 1D iadst (4x4 at the FULL
  ±2^15 spec input bound), the SB/SH ops at the exact spec T bound over every network angle and
  both flips, and the fused 2D vs scalar for all three tx types x 4/8/16. Perf (A/B, same
  session): 854x356 inter movie 51.52 -> 51.83 MP/s (+0.6%, within noise, as the ~0.3% estimate
  predicted); on `vp90-2-16-intra-only` the `InverseTransform` sub-timer went 3.7 ms -> 2.6 ms
  (10.7% -> 7.7% of decode) -- directional only (7-frame clip).
- Scaled-reference (SVC / resize) inter-prediction SIMD: DONE 2026-07-24
  (`simd.rs::block_inter_predict_scaled_avx2`; the scalar loops moved to
  `predict::block_inter_predict_scalar`, the always-kept fallback + unit-test oracle). Both FIR
  passes are AVX2, all bit depths (same i32-FIR + `max_val` argument as the unscaled kernels):
  the horizontal pass's per-column subpel phase and source column (`p = x + x_step*c`) are
  row-invariant, so they are precomputed once per call (per-column gather indices + tap-major
  coefficient vectors) and each tap's 8 samples fetched with `_mm256_i32gather_epi32` from a
  per-row i32 scratch of the edge-clamped source span; the vertical pass's phase/base row are
  uniform per output row (no gathers -- the unscaled vertical pass with a per-row filter).
  Edge clamping happens inside the kernel via the precomputed clamped indices (bit-identical to
  the scalar border replication), so unlike the unscaled kernels there is no `in_bounds` scalar
  fallback; the dispatch's two `MAX_INTERMEDIATE_HEIGHT` bound checks are safety guards against
  non-conformant scaling ratios (see the landmine in `implementation-notes.md`). Bit-exact
  (sweep 315/315 both SIMD configs -- incl. the 29 MD5-gated resize/SVC vectors:
  `vp90-2-05-resize`, `vp90-2-13-largescaling`, `vp90-2-18-resize`, the 24
  `vp90-2-21-resize_inter_*`, `vp90-2-22-svc_1280x720_{1,3}`) + ffmpeg cross-decode; a
  temporary probe confirmed the kernel is actually exercised by the 05/13/18/21-resize and
  3-layer-SVC vectors (the `resize-fp-tiles` family resizes only at keyframes, so it decodes
  without scaled inter refs); a unit test pins the kernel against the scalar across widths
  (4-pad + 8-wide groups), steps 1..=32 on both axes, subpel phases, all edge clamps, and
  8/10/12-bit. Perf (A/B via a temporary kernel-only gate, same binary): on the most
  scaled-heavy official clip, `vp90-2-22-svc_1280x720_3` (3-layer SVC, scaled inter-layer refs
  every frame), **42.4 -> 47.2 MP/s (min), ~+11%**; `InterPredict` sub-timer 172 -> 134 ms
  (37.6% -> 32.4% of decode). `vp90-2-21-resize_inter_1280x720_5_1-2` (resizes only every few
  frames, mostly unscaled) is within noise, as expected. No large scaled-content clip exists in
  the corpus (the SVC clip is 60 frames), so the numbers are directional.
- Wave 4 (remaining): intra prediction is the only remaining named hot spot, but profiled at
  ~0.3% on inter content -- not worth SIMD unless targeting intra-heavy / all-intra streams.
  Still scalar (perf gap only): the WHT (lossless 4x4) inverse transform, 10/12-bit ADST, and
  the unscaled inter path's near-reference-edge blocks (the `in_bounds` scalar fallback; the
  scaled kernel's gather-through-clamped-scratch approach could close it if ever profiled as
  hot). Inter-pred (unscaled + scaled), the loop filter, and the DCT_DCT transform are SIMD at
  all bit depths; the ADST-containing transforms at 8-bit.
- Tile-parallel multithreading (a different lever than SIMD, same realtime goal): DONE
  2026-07-23 (`tile::decode_tiles_parallel` / `spawn_column_worker` / `merge_column_worker` +
  `Counts::add_assign`). A frame with >1 tile column and 1 tile row decodes each column on its own
  worker `TileDecoder` via `std::thread::scope` (no external crate), then merges the disjoint
  column strips + sums the per-worker counts. Bit-exact (sweep 315/315 both configs, incl. the
  `vp90-2-08-tile_1x{2,4,8}` vectors). **51.58 -> 62.87 MP/s, 1.22x** on a 2-column 854x356 clip;
  scales with tile count. `tile_rows > 1` and single-column frames stay sequential. Columns are
  decoded in chunks of `available_parallelism()` so thread count and per-worker fixed overhead are
  bounded (not the attacker-declarable `tile_cols`); the `robustness_test` fuzz corpus
  includes a 4-tile clip so the parallel path is fuzzed for no-panic. Follow-up (cut the
  per-worker buffer waste): DONE 2026-07-24 -- chose the column-width-buffer design (safe code):
  each worker's planes/`mi_grid` are column STRIPS with an origin (`Plane::new_strip` /
  `MiGrid::new_strip`; accessors keep taking absolute coordinates, so the decode path is
  untouched -- only the fused SIMD reconstruction's raw-slice offset translates explicitly, and
  the last column's strip extends into the superblock-rounded padding its edge blocks write).
  Per-worker allocation drops from a full frame to ~`frame/tile_cols` (1920x800 4-tile: 6.9 MiB
  -> 1.7 MiB per worker; all workers sum to ~1 frame regardless of tile count); the merge copy
  remains (unchanged bytes, now read from compact strips). The shared-buffer (unsafe
  disjoint-write) variant was REJECTED: strips are row-interleaved, so disjoint `&mut` handout is
  impossible and soundness would need raw-pointer/UnsafeCell plumbing through the hottest write
  paths -- for a merge memcpy worth ~1% of decode. Bit-exact (sweep 315/315 both SIMD configs;
  new `tests/tile_parallel_test.rs` pins parallel == forced-sequential byte-for-byte on
  `vp90-2-08-tile_1x{2,4,8}` via the test-only `tile::FORCE_SEQUENTIAL_TILES` knob). Perf
  (interleaved A/B vs HEAD, same session): 1920x800 4-tile **~78 -> ~91 MP/s (min), ~+17%** (the
  per-frame alloc+zero of 4 full frame+grid buffers, ~27 MiB/frame, dominated); 854x356 2-tile
  ~parity (+~1%, within this machine's noise).
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
- **Adversarial 4-lens review of the four 2026-07 pending changes** (HBD DCT / 8-bit ADST /
  scaled inter-pred AVX2; tile-parallel strip buffers): DONE 2026-07-24. No confirmed bug in
  the changes themselves. Fixed alongside: a PRE-EXISTING scaled-path panic on malformed
  ratios (now rejected per block, `TileError::RefFrameSizeOutOfRange` + red→green
  `tests/synthetic_scaled_ref_test.rs`), a resize seed for the robustness fuzz, stale
  SIMD/tile-parallel docs, and test hardening (fused reconstruction at nonzero origins;
  `b_op` at spec bounds). Recorded-but-not-done ideas: (1) a differential fuzz mode (SIMD vs
  `VP9DEC_NO_SIMD` on a malformed corpus — needs the caveat that i32-wrap divergence on
  garbage input is legitimate, see implementation-notes); (2) a synthetic wide (≥449px)
  HBD/profile-2 multi-tile stream to exercise the corpus-unreachable {HBD, non-4:2:0} ×
  column-strip cells (currently sound-by-construction + offset unit tests only).

## Non-goals (decided, not deferred)

- Consulting libvpx / other decoder SOURCE for decode logic (clean-room rule; ffmpeg
  OUTPUT comparison as an oracle is fine and established).
