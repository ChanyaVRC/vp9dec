# Implementation notes

Still-relevant, non-obvious design rationale, landmines, and known gaps — the things that
aren't clear from the code, the spec, or `README.md` (which holds the architecture and module
map). **Keep this concise.** Once an issue is resolved and the current behavior is captured
here, the blow-by-blow (root-cause analyses, per-change verification logs, the wave-by-wave
build narrative) belongs in the git log; `docs/history.md` keeps only a concise milestone
overview. A resolved bug needs no entry.

## Current state

- **Profiles 0-3.** 8/10/12-bit depth and 4:2:0 / 4:2:2 / 4:4:0 / 4:4:4 subsampling. Output is
  `Frame { bit_depth, subsampling_x/y, y/u/v: PlaneData }`, where `PlaneData` is `U8` for 8-bit
  and `U16` for 10/12-bit. The full official sweep is 315/315 bit-exact, in both SIMD configs.
- **Segmentation.** Full: segment-id decode (including temporal prediction) and all four
  `SEG_LVL_*` features. Official vectors exercise seg-id + `SEG_LVL_ALT_Q`; the other three
  (`ALT_L` / `REF_FRAME` / `SKIP`) have no official vector — see "Known gaps".
- **Superframes.** `Decoder::decode_frame(chunk)` splits a superframe internally and returns one
  `DecodedFrame` per constituent VP9 frame. A hidden frame (`show_frame == 0`) has `frame:
  None`; a `show_existing_frame` chunk has `info: None`. When one chunk shows more than one
  constituent (3-layer SVC), only the **last** shown one is display output (matches the
  libvpx/ffmpeg oracle).
- **Cross-frame ownership.** DPB slots, `prev_mi_grid` / `prev_segment_ids`, and the compressed-
  header probability context are shared via `Arc` or borrowed, not deep-cloned per frame
  (performance only — the sweep is byte-identical before/after).
- **SIMD.** AVX2 covers, for **all bit depths**: inter-prediction -- unscaled (widths 8/16/32/64
  via `block_inter_predict_avx2`, width 4 via the 128-bit `block_inter_predict_avx2_w4`) and
  the general edge-clamping path (`block_inter_predict_scaled_avx2`), used both for reference-
  scaled prediction (SVC / resize) and for unscaled blocks whose filter window crosses a
  reference edge: the horizontal pass's per-column subpel phase + source column are row-
  invariant, so they are precomputed per call and each tap's samples fetched by
  `_mm256_i32gather_epi32` from a per-row i32 scratch of the edge-clamped source span; the
  vertical pass's phase/base row are uniform per output row -- no gathers; edge clamping happens
  inside the kernel, so no scalar border fallback; the subpel
  FIR is bit-depth-agnostic and its i32 accumulation holds a 12-bit sample, so only the `clip1`
  bound differs -- the caller passes `max_val = (1<<bit_depth)-1`; loop-filter edges on
  **both** passes -- horizontal (`loop_filter_horiz8_avx2`) and vertical (`loop_filter_vert8_avx2`,
  which transposes the tap window into the horizontal kernel's layout and reuses it), each
  covering narrow / wide8 / wide16 (the 8-bit base / clamp / flat-threshold constants scale by
  `<< (bit_depth-8)` inside the kernel); and the **DCT_DCT inverse transform + reconstruction**
  (all sizes 4/8/16/32 -- the scalar recursive idct mirrored on i32 8-lane vectors, fused with
  the residual-add + `(1<<bit_depth)-1` clip so the result is written straight into the plane,
  skipping the i64 round-trip; 8-bit takes `inverse_transform_dct_dct_reconstruct_avx2`, 10/12-bit
  the `_hbd_` variant whose butterfly products are widened to i64 -- see the landmine below);
  and the **ADST-containing transforms** (ADST_DCT / DCT_ADST / ADST_ADST, sizes 4/8/16 --
  32x32 is DCT-only): 8-bit on the i32-lane networks (`inverse_transform_adst_reconstruct_avx2`,
  same fused reconstruction), 10/12-bit on i64-lane networks
  (`inverse_transform_adst_reconstruct_hbd_avx2`, 4 i64 lanes per vector -- the ADST's unrounded
  `S` array needs `24 + BitDepth` bits of lane storage, and both axes of a mixed block run in
  one self-contained i64 driver; see the landmine below). Among the transforms only WHT
  (lossless) stays scalar.
  Runtime-detected and cached; `VP9DEC_NO_SIMD=1` forces scalar. Output must equal the scalar
  path — the sweep passes 315/315 in both configs. Routing unscaled reference-edge blocks
  through the general kernel cut their measured stage time by 2.66-2.90x and improved whole-
  clip throughput by 28-35% on representative 426p/1080p content. A focused width-4 edge
  benchmark covered 8/10/12-bit, heights 4/8, and all four borders: all 24 cells were faster
  than scalar (1.21x minimum, 1.38x aggregate), so the padded 8-lane gather is still a win for
  the narrowest blocks.
- **Tile-parallel decode.** A frame with >1 tile column and exactly 1 tile row decodes each tile
  column on its own worker `TileDecoder` (`std::thread::scope`, no external crate), then merges
  the disjoint column strips (planes + mi_grid) and sums the per-worker adaptation counts back
  into one. Each worker's planes and `mi_grid` are **column-strip buffers** sized to its tile
  (`Plane::new_strip` / `MiGrid::new_strip`, ~`frame/tile_cols` each, summing to ~1 frame across
  workers): they carry the strip's absolute origin and their accessors keep taking absolute
  frame coordinates, so the decode path is identical whole-frame vs strip. Bit-exact with
  sequential decode (the sweep's `vp90-2-08-tile_1x{2,4,8}` vectors pass in both SIMD configs;
  `tests/tile_parallel_test.rs` additionally pins parallel == forced-sequential byte-for-byte).
  `tile_rows > 1` (above-context crosses tile-row boundaries) and single-column frames stay
  sequential. ~+22% on a 2-column 854x356 clip at introduction; the strip buffers added ~+17%
  on a 4-tile 1920x800 clip (per-frame worker alloc+zero shrank ~4x).
- **Loop-filter parallelism.** The deblocking filter (once per frame, after tile decode joins) runs
  all three planes in ONE `thread::scope` on a WAVEFRONT of worker threads
  (`loop_filter::wavefront_filter_planes`, no external crate): superblock rows round-robin across
  `min(available_parallelism, sb_rows)` workers, each row lagging the one above by 2 superblocks
  (see the landmine). Workers flow across the independent planes with no per-plane barrier (each
  plane has its own progress array). Workers write disjoint pixels of one plane buffer through an
  `unsafe` raw `PlaneView`; the sequential / small-frame (`LF_PARALLEL_MIN_MI`) path uses `Plane`
  directly and is fully safe — both behind the `PlaneAccess` trait so the filter arithmetic is
  written once, and the AVX2 kernels take a `*mut u16` base. Bit-exact (sweep both configs +
  `tests/loop_filter_parallel_test.rs` pins wavefront == forced-sequential over repeated iters). The
  loop filter went from the serial bottleneck (~38–52% of decode) to 11–18%; ~÷6.8 vs a
  single-threaded loop filter, so 1080p decodes at ~98 MP/s (single-tile) / ~155 MP/s (4-tile). A
  persistent cross-frame pool was evaluated and rejected (the residual per-frame spawn is ~0.5% of
  decode, not worth the `unsafe` lifetime erasure it needs).
- **Generated robustness / differential coverage.** `tests/structured_fuzz_test.rs` preserves
  uncompressed headers, chunk/superframe/tile boundaries, and valid prefix frames while mutating
  compressed-header or tile entropy suffixes; `VP9DEC_FUZZ_LONG_ITERS` enables a bounded extended
  campaign. `tests/simd_scalar_differential_test.rs` starts isolated SIMD and forced-scalar
  processes and compares the exact input plus decoded output of 27 conformant generated
  scenarios spanning profiles, bit depths, subsampling, segmentation, HBD tiles, and 2x scaling.

Architecture and the module map live in `README.md`; the acceptance gate is the `verify-vp9dec`
skill; change-navigation is the `vp9dec-architecture` skill.

## Landmines (bit-exactness hazards — read before touching these)

- **Trust the empirical sweep over spec / static reasoning.** The sub-8x8 chroma MV block index
  in `tile/residual.rs` is `(y * num4x4w + x)` and is bit-exact-correct. A plausible,
  spec-grounded "correction" to it (meant to fix 4:2:2) once *regressed* the official 4:2:2
  vector. When a hypothesis and the sweep disagree, the sweep wins — verify first, edit second.
- **Tile-parallel decode (`decode_tiles_parallel` in `src/tile/parallel.rs`) assumes tile-column
  independence -- and the worker buffers are now sized to that assumption.** Each tile column decodes into its own
  worker `TileDecoder` whose planes/`mi_grid` are column STRIPS covering only its tile's columns
  (absolute-coordinate accessors with an internal origin), merged back by disjoint column strip.
  This works only because every current-frame access a tile decode makes stays within its own
  columns: VP9 gates the left neighbor at the tile boundary (`tile.rs`'s
  `avail_l = col > mi_col_start`), intra above-right reads are bounded by the prediction block
  (`not_on_right`), MV candidates by `is_inside` (tile-bounded), and no block reads the
  not-yet-decoded column to its right; the one deliberate exception is that edge superblocks of
  the LAST tile column write past `MiCols*8` into the superblock-rounded padding, so that
  column's strip extends to the full padded width (`spawn_column_worker`). A change that lets a
  block read or write across a tile-column boundary now panics on the strip bounds (debug) or
  corrupts the parallel path (release) -- re-run the `vp90-2-08-tile_1x{2,4,8}` sweep vectors AND
  `tests/tile_parallel_test.rs` after touching availability or prediction near tile edges. The
  merge copies only the pixel columns up to `mi_col_end*8`; padding past `mi_cols` is never
  copied, so it stays zero in the merged frame exactly as in a sequential decode.
- **The loop-filter wavefront (`loop_filter::wavefront_filter_planes`) is `unsafe`, and its
  2-superblock row lag is load-bearing.** Worker threads write DISJOINT pixels of one plane buffer
  concurrently through a shared raw `PlaneView` (Rust forbids the aliasing `&mut` this would need),
  so nothing checks that the accesses are actually disjoint — correctness rests entirely on the
  wavefront ordering. Superblock (r,c) may be filtered only once row r-1 has finished column
  **c+2**, not c+1: (r,c)'s horizontal (top-edge) pass and (r-1,c+1)'s vertical (left-edge) pass
  both write a shared 8x8 corner where the superblocks meet, so a 1-column lead would let the row
  above still be writing that corner. The gate is a per-row `AtomicU32` progress counter with
  `Release` on publish / `Acquire` on wait (which also publishes a worker's written pixels before
  the row below reads them at the boundary). Get the lag or the ordering wrong and you get an
  INTERMITTENT race — it may pass a sweep once and fail later. After any change here, re-run
  `tests/loop_filter_parallel_test.rs` (wavefront == forced-sequential, repeated) AND the full sweep
  in both SIMD configs several times; a single green run is not enough to trust a concurrency change.
- **All four AVX2 kernel families (inter-pred, loop filter, DCT_DCT transform, ADST-containing
  transforms) are all-depth, each by a different mechanism -- and the ADST's per-depth kernel
  SPLIT is load-bearing.** Inter-pred takes a `max_val` clip bound, the loop filter scales its
  8-bit constants by `<< (bit_depth-8)`, the DCT_DCT transform swaps its butterfly multiply
  width per depth (next landmine), and the ADST-containing transforms swap the whole LANE WIDTH:
  spec §8.7.1.1 bounds the ADST's unrounded `S` array to signed `24 + BitDepth` (up to 36)
  bits -- beyond i32 lane *storage* at 10/12-bit, not merely the products -- so the DCT's
  products-only i64 widening does not carry over there. The i32-lane ADST kernels
  (`inverse_transform_adst_reconstruct_avx2` / `iadst*_simd` / `sb_op_simd` / `sh_op_simd`)
  MUST stay dispatched only at `bit_depth == 8`; 10/12-bit routes to the i64-lane networks
  (`inverse_transform_adst_reconstruct_hbd_avx2` / `iadst*_i64` / `idct_i64`, 4 i64 lanes per
  vector, both axes of a mixed block in i64), whose exactness argument is different in kind:
  they are the scalar's i64 arithmetic op for op (exact-mod-2^64 multiply, emulated 64-bit
  arithmetic-shift `round2`), narrowing to i32 only after the final `round2` where spec
  §8.7.1.1 re-bounds the value. Any *new* SIMD kernel with hardcoded 8-bit constants or
  i32-tight arithmetic must likewise gate on `bit_depth == 8` until it is made depth-aware.
- **The AVX2 DCT_DCT inverse transform stores lanes as i32 at every depth; only the butterfly
  multiplies are depth-gated.** Spec §8.7.1.1 bounds every stored transform intermediate to
  signed `8 + BitDepth` bits (<= 20 at 12-bit) -- that is why i32 lanes are safe at all depths --
  but the `t*cos64` products are not: they fit i32 only at 8-bit (16b*15b). The dispatch
  (`tile/residual.rs`) routes `bit_depth == 8` to the `mullo`-product kernel
  (`inverse_transform_dct_dct_reconstruct_avx2`) and 10/12-bit to the `_hbd_avx2` variant, whose
  `b_op_simd_hbd` computes the products with 32x32->64-bit widening multiplies
  (`_mm256_mul_epi32`) and rounds in i64 (its `>> 14` is a LOGICAL 64-bit shift -- exact because
  only the low 32 bits are kept and logical/arithmetic shifts agree there). Do not route 10/12-bit
  to the 8-bit kernel or reuse the i32 `mullo` butterfly in depth-general code. WHT (lossless)
  stays scalar. On NON-conformant streams (coefficients past spec §8.7.1.1's
  stored-value bounds) the SIMD i32 lanes may wrap where the scalar i64 arithmetic does not
  (likewise the i64-lane ADST driver's final i32 narrowing), so
  SIMD-on vs `VP9DEC_NO_SIMD=1` outputs can legitimately diverge on garbage input -- a
  differential fuzzer comparing the two configs on a malformed corpus would misread that as a
  SIMD bug; only conformant streams are comparable. The generated differential test above
  deliberately emits skipped/lossless conformant blocks, so malformed coefficient overflow
  cannot contaminate its oracle.
- **The edge-clamping inter-pred dispatch's two bound checks (`predict.rs`: `intermediate_height` and
  the horizontal `span`, both `<= MAX_INTERMEDIATE_HEIGHT`) are safety guards for the unsafe
  kernel's fixed scratch -- redundant by construction since the per-block ratio rejection, but
  keep them.** Out-of-range scaling ratios never reach any predict path anymore: `decode_block`
  rejects them per block (`TileError::RefFrameSizeOutOfRange`, see below). Note the scalar
  fallback's own fixed scratch has the identical steps-`<= 32` limit and panics on its slice
  bounds past it, so "malformed streams fall back to scalar" was never a safe handler -- the
  upstream rejection, not the fallback, is what makes out-of-range input safe.
- **Loop-filter constants are 8-bit and scale by `<< (bit_depth - 8)`** (identity at 8-bit). Any
  loop-filter change must be re-verified at 10/12-bit, not just 8-bit.
- **The whole per-frame loop filter is gated on the frame-level `loop_filter_level` (spec §8.1),
  before per-block ref/mode deltas apply.** Those deltas can raise a block's level above 0, so
  running the filter on a frame that signaled "no filtering" is wrong.
- **`decode_block` rejects three malformed-bitstream conditions before the residual/predict path**
  (which indexes fixed size/tx tables and unwraps reference views without re-checking): a
  `BLOCK_INVALID` chroma block size, an inter block referencing an empty DPB slot, and an inter
  block referencing a frame more than 2x the current frame's width or height
  (`TileError::RefFrameSizeOutOfRange`: spec §8.5.2.3's conformance bound, which caps the
  scaling steps at 32 and thereby sizes every motion-compensation scratch --
  `predict::MAX_INTERMEDIATE_HEIGHT`; past it the scaled scalar path panics on its slice
  bounds, as a 64x256-keyframe -> 64x64-inter stream used to). The size check MUST stay
  per-block, not at reference-resolution time: a conformant stream may *list* an out-of-range
  slot in `ref_frame_idx` without predicting from it (`vp90-2-22-svc_1280x720_3`'s base layer
  lists the 4x-larger enhancement layer; an eager check broke it). Conformant streams never
  trip these, so they change no valid-input output -- don't remove them as "dead" checks.
  Regression-guarded by `tests/robustness_test.rs` (a fuzz for no-panic-on-bad-input) and
  `tests/synthetic_scaled_ref_test.rs` (4x ratio rejected; exactly-2x decodes).
- **Malformed-input rejection checks are conformance-safe but empirically tuned -- don't loosen
  them without re-running the full sweep AND `tests/invalid_vector_test.rs`.** To pass the
  official `invalid-*` gate the decoder rejects, before/without emitting garbage: an absurd frame
  size (`HeaderError::FrameSizeTooLarge`, cap `MAX_FRAME_LUMA_SAMPLES`, a DoS guard against an
  8 GB allocation abort), an inter reference whose bit-depth/subsampling differs from the current
  frame (`RefFrameFormatMismatch`), an inter block predicting from a reference beyond the 2x
  scaling bound (`TileError::RefFrameSizeOutOfRange`, per block -- see above), and a tile whose
  arithmetic decoder finishes far off its
  buffer end -- either over-reading past it or leaving a large unused tail
  (`TileError::CorruptTile`, thresholds `TILE_OVER_READ_LIMIT_BITS` / `TILE_UNDER_READ_LIMIT_BITS`
  in `tile.rs`). The tile thresholds sit ~70x above the largest slack any of the 315 conformant
  vectors leaves (measured: ≤14 unused bits, 0 over-read) and ~10x below the smallest corruption
  the invalid corpus shows, so valid output is untouched -- but they are heuristic margins, not
  spec constants.

## Conventions worth knowing

- **10/12-bit conformance MD5** is computed over the output as **16-bit little-endian** (libvpx
  convention); `tests/common`'s `i420_bytes` emits `U16` planes that way so the hashes match.
- **`cargo fmt` is safe tree-wide.** `rustfmt <one file>` on the crate root used to reformat the
  whole crate; the tree was normalized once, so a plain `cargo fmt` now touches only your change.
- **Unit-test bodies live under `tests/unit/`**, included from their owning source module with
  `#[cfg(test)] #[path = "..."] mod tests;`. They therefore retain private-module access without
  widening the decoder API, while `src/` remains production implementation. Shared unit fixtures
  live in `tests/unit/support.rs`; Cargo integration tests remain the top-level `tests/*.rs` files.

## Known gaps

- **Deliberately scalar / target-conditional SIMD paths.** Intra prediction and the lossless
  4x4 WHT remain scalar; aarch64 uses the same scalar fallback and has no NEON mirror. The x86
  measurement gate found intra prediction at 0.3-1.7% on representative inter content and 3.1%
  (0.9 ms total) on the short intra-only vector. The WHT-containing inverse-transform stage was
  7.9% (1.6 ms total) on a two-frame lossless vector, but is absent from ordinary lossy content.
  These do not justify more architecture-specific code for the current workload. Revisit only
  for a measured intra/lossless workload or a concrete aarch64 deployment target; neither is
  open work today.
- **`SEG_LVL_ALT_L` / `SEG_LVL_REF_FRAME` / `SEG_LVL_SKIP` have no official test vector.** They
  are proven instead by synthetic round-trip vectors (`tests/synthetic_seg_test.rs`)
  cross-decoded byte-identically by ffmpeg's `libvpx-vp9` and native `vp9` decoders.
- **19 corpus clips ship no upstream `.md5`** (`vp90-2-bbb_*` / `vp90-2-tos_*` /
  `vp90-2-sintel_*` movies), so the sweep can't MD5-check them and they're excluded from the 315.
  Instead they're cross-checked against ffmpeg's `libvpx-vp9`: the 12 tos/sintel clips in full
  (268,832 displayed frames byte-identical), and the 7 bbb clips as a spot-check — first 1000
  frames each, 7000 total, byte-identical (`tests/bbb_cross_check.rs`).
