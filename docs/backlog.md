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
  configs (full sweep 315/315) + ffmpeg cross-decode; a 500-trial `tests/unit/loop_filter.rs` unit
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
- 10/12-bit ADST inverse-transform SIMD (ADST_DCT / DCT_ADST / ADST_ADST, sizes 4/8/16):
  DONE 2026-07-24 (`simd/transform.rs::inverse_transform_adst_reconstruct_hbd_avx2` +
  `iadst4/8/16_i64` / `idct_i64` / `sb_op_i64` / `sh_op_i64` / `b_op_i64` / `mul_i64` /
  `round2_i64`; driver `xform_pass_i64`). The last non-lossless transform gap. Design: i64
  LANES (4 per `__m256i`, 4 rows per pass chunk) -- the spec §8.7.1.1 bound on the ADST's
  unrounded `S` array is `24 + BitDepth` (up to 36) bits, beyond i32 lane STORAGE, so the HBD
  DCT's products-only widening could not carry over. Both axes of a mixed block run in i64
  (the i64 1D DCT is trivially bit-exact -- scalar `idct` IS i64 -- and keeps the driver
  self-contained, no i32/i64 layout mixing across the transpose). Every op mirrors the scalar
  i64 arithmetic exactly: multiply-by-constant via an exact-mod-2^64 low/cross-product
  decomposition (AVX2 has no 64-bit multiply), `round2` via an emulated 64-bit arithmetic
  shift; the final post-round2 outputs (re-bounded by §8.7.1.1) narrow to i32 into the shared
  `reconstruct_add_clip` with the `(1<<bit_depth)-1` clip. Dispatch (`tile/residual.rs`) is
  now simply `!lossless && avx2_enabled()`; WHT/lossless stays scalar and `VP9DEC_NO_SIMD=1`
  still forces scalar everywhere. Bit-exact (sweep 315/315 both SIMD configs, all eight
  `vp92/vp93-2-20-*` 10/12-bit vectors PASS in both) + ffmpeg cross-decode 10/10 + robustness
  fuzz 6/6 seeds 0 panics (10-bit seed exercises the path under corruption) + invalid gate
  21/21. A temporary probe confirmed the official corpus exercises ALL 18 cells (2 depths x 3
  tx types x 3 sizes) -- e.g. AdstAdst 16x16 at 12-bit via `vp93-2-20-12bit-yuv440`. Unit
  tests pin the 1D iadst AND the mixed-axis 1D idct at the FULL ±2^(7+BitDepth) spec input
  bound for 10-bit and 12-bit (possible only because the lanes are i64 -- the scalar i64 path
  is the oracle), with a self-check that those magnitudes DIVERGE on the i32-lane network
  (proving the i64 lanes are load-bearing), and the fused 2D vs scalar for all type/size/depth
  cells including a nonzero-origin/wider-stride placement. Perf: completeness item, as
  expected -- the only HBD corpus clips are 160x90x10-frame (~8 ms/decode); whole-clip MP/s is
  within noise and the `InverseTransform` sub-timer reads 0.6 ms before AND after on the
  12-bit clip (below the 0.1 ms display resolution -- no measurable delta; the HBD DCT change
  had already taken the DCT_DCT share). The 8-bit and HBD-DCT kernels are untouched (separate
  fns; 854x356 8-bit clip re-benched at par).
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
  Still scalar (perf gap only): the WHT (lossless 4x4) inverse transform and
  the unscaled inter path's near-reference-edge blocks (the `in_bounds` scalar fallback; the
  scaled kernel's gather-through-clamped-scratch approach could close it if ever profiled as
  hot). See implementation-notes.md for current SIMD coverage.
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
- Loop-filter plane-parallel (a third realtime lever, after SIMD and tile-parallel): DONE
  2026-07-25 (`loop_filter::loop_filter_frame` / `loop_filter_plane`). The deblocking filter runs
  once per frame on the main thread AFTER the tile decode joins, so on multi-tile HD it had become
  the serial tail -- profiled at 38.5% of single-tile COMPUTE and 52% of 4-tile WALL-CLOCK (both
  big compute stages, InterPredict and LoopFilter, are already AVX2, so this is a threading lever,
  not a new kernel). The three plane buffers are disjoint and never read one another (each
  superblock filter touches only its own plane; `mi_grid`/`lvl_lookup` are shared read-only), so
  the plane loop is hoisted OUTERMOST and the planes filtered on separate `std::thread::scope`
  threads (luma on the caller, chroma spawned) -- bit-exact by construction (each plane's
  (row,col,pass) raster order per spec §8.8 is preserved exactly; only the interleaving between the
  independent planes changes). `superblock_loop_filter`/`_edge_avx2` refactored to take a single
  `&mut Plane`; size-gated (`LF_PARALLEL_MIN_MI`, sub-VGA stays sequential). Bit-exact (sweep
  315/315 both SIMD configs + ffmpeg cross-decode 10/10 + full suite). Perf (interleaved A/B, same
  session, 32-core): 1080p single-tile **52.3 -> 60.1 MP/s (+15%)**, 4-tile **66.6 -> 97.3 MP/s
  (+46%)**; LoopFilter stage ÷1.58 / ÷1.90 (chroma is proportionally more expensive than its pixel
  count, so the 3-way split balances better than the ~1.5x Y-bound estimate). Follow-up
  (intra-plane luma wavefront): DONE 2026-07-25 (`wavefront_filter_plane` + `PlaneView` +
  `PlaneAccess`). Each plane is now filtered by a WAVEFRONT of worker threads (superblock rows
  round-robin across `min(available_parallelism, sb_rows)` workers), replacing the concurrent-plane
  split so luma AND chroma each get the full pool. A worker filters superblock (r,c) only after row
  r-1 reached column c+2 -- a 2-superblock lag, because (r,c)'s top-edge (horizontal) pass and
  (r-1,c+1)'s left-edge (vertical) pass write a shared 8x8 corner; a 1-column lag would race it.
  The workers write DISJOINT pixels of one plane buffer through a shared `unsafe` raw view
  (`PlaneView`, Send+Sync); Rust forbids the aliasing `&mut` this would need, so soundness rests on
  the wavefront ordering plus `Release`/`Acquire` on a per-row progress counter. The sequential path
  stays fully SAFE: the filter arithmetic is generic over a `PlaneAccess` trait (`Plane` = safe,
  `PlaneView` = raw), and the AVX2 kernels take a `*mut u16` base. Landed in two verified steps: (2a)
  the raw-access refactor alone, single-threaded (sweep 315/315 both configs -- isolating raw-access
  correctness from any race), then (2b) the wavefront. Verified: new
  `tests/loop_filter_parallel_test.rs` decodes HD clips with the wavefront and with a test-only
  `FORCE_SEQUENTIAL_LOOP_FILTER` knob, asserting byte-identical over 5 parallel iters (617 frames x
  5 all == the sequential reference); sweep 315/315 x3 (SIMD-on x2 + forced-scalar) + ffmpeg
  cross-decode 10/10 + clippy/fmt. Perf (interleaved A/B vs the plane-parallel Phase above, same
  session, 32-core): 1080p single-tile **78.7 -> 92.2 MP/s (+17%)**, 4-tile **111.9 -> 142.1 MP/s
  (+27%)**; LoopFilter stage ÷1.71 / ÷1.81, below the wavefront's latency-bound ceiling.
  Further follow-up (fuse the per-plane scopes): DONE 2026-07-25 (`wavefront_filter_planes`). The
  three planes were filtered in three separate `thread::scope`s, so all workers hard-barriered twice
  per frame (each plane fully drained before the next began). Fusing them into ONE scope whose
  workers flow across the independent planes removed both drains (and cut per-frame spawns 3x) --
  the drain, not the spawns, was the dominant loss. LoopFilter stage a further ÷1.74-1.80
  (single-tile 1309 -> 726 ms), **+7% single-tile / +11% 4-tile** on top of the wavefront, all still
  SAFE `thread::scope`. Final state: the loop filter is now 11-18% of decode (~÷6.8 vs
  true-sequential, ~83% of the latency-bound ceiling); 1080p single-tile ~98 MP/s, 4-tile ~155 MP/s.
  A persistent CROSS-FRAME thread pool was then evaluated and REJECTED: the only thing left to
  recover is the per-frame spawn (~17 workers x ~10us ~= 0.5% of decode), which does not justify the
  `unsafe` lifetime erasure + global shared-pool concurrency such a pool would need.
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

- **Structural refactor pass (audit follow-up)**: DONE 2026-07-24. Pure structural moves on the
  decode path, no output change (full acceptance gate re-verified, both SIMD configs): tile-
  parallel machinery extracted to `src/tile/parallel.rs` (R1); the duplicated 4-byte tile-size
  parse and corrupt-tile check unified into `split_tiles`/`check_tile_read_bounds` (R2, the one
  approved behavior-adjacent change -- on some malformed multi-tile streams the error VARIANT
  may differ, still `Err` from the same packet; invalid gate stays 21/21); `residual()`'s
  per-plane inter-predict block and `tokens_and_reconstruct`'s dequant/transform tail split out
  at spec-process seams (R3); `src/simd.rs` (2,595 lines) split into a hub +
  `simd/{inter, loop_filter, transform}.rs` plus `tests/unit/simd.rs` (S1/S2); the twin loop-filter
  AVX2 edge dispatchers merged into one pass-parameterized `superblock_loop_filter_edge_avx2` (S3);
  `refresh_probs` and the PrevSegmentIds refresh extracted from `decode_one_frame` (A5). `fetch-vectors.sh` captured `rc=$?` after
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
- **Out-of-line unit-test files**: DONE 2026-07-25. All unit-test bodies now live under
  `tests/unit/` and are included from their owning module with `#[cfg(test)] #[path = "..."]`.
  This preserves private access without widening the decoder API and keeps `src/` focused on
  production/test-support implementation. Convention recorded in `implementation-notes.md`.
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
