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
- **SIMD.** AVX2 covers, for 8-bit content: unscaled inter-prediction (widths 8/16/32/64 via
  `block_inter_predict_avx2`, width 4 via the 128-bit `block_inter_predict_avx2_w4`); loop-filter
  edges on **both** passes -- horizontal (`loop_filter_horiz8_avx2`) and vertical
  (`loop_filter_vert8_avx2`, which transposes the tap window into the horizontal kernel's layout
  and reuses it), each covering narrow / wide8 / wide16; and the **DCT_DCT inverse transform +
  reconstruction** (all sizes 4/8/16/32, `inverse_transform_dct_dct_reconstruct_avx2` -- the
  scalar recursive idct mirrored on i32 8-lane vectors, fused with the residual-add + 8-bit clip
  so the result is written straight into the plane, skipping the i64 round-trip). Runtime-detected
  and cached; `VP9DEC_NO_SIMD=1` forces scalar. Output must equal the scalar path — the sweep
  passes 315/315 in both configs.

Architecture and the module map live in `README.md`; the acceptance gate is the `verify-vp9dec`
skill; change-navigation is the `vp9dec-architecture` skill.

## Landmines (bit-exactness hazards — read before touching these)

- **Trust the empirical sweep over spec / static reasoning.** The sub-8x8 chroma MV block index
  in `tile/residual.rs` is `(y * num4x4w + x)` and is bit-exact-correct. A plausible,
  spec-grounded "correction" to it (meant to fix 4:2:2) once *regressed* the official 4:2:2
  vector. When a hypothesis and the sweep disagree, the sweep wins — verify first, edit second.
- **The loop filter's AVX2 dispatch is gated on `bit_depth == 8`.** The AVX2 kernels
  (`src/simd.rs`) are u8-only; letting them run on a 10/12-bit path silently corrupts output.
  The inter-prediction AVX2 path is gated the same way. Any new SIMD kernel must gate likewise.
- **The AVX2 DCT_DCT inverse transform uses i32 lane storage, valid only at `bit_depth == 8`.**
  Spec §8.7.1.1 bounds 8-bit transform intermediates to 16 bits, so every `t*cos64` product fits
  i32; at 10/12-bit they would overflow. The dispatch (`tile/residual.rs`) gates on
  `bit_depth == 8 && tx_type == DctDct && !lossless` -- do not widen it. ADST / WHT / mixed
  transforms stay scalar (they are the minority of blocks and outside the i32-safe assumption).
- **Loop-filter constants are 8-bit and scale by `<< (bit_depth - 8)`** (identity at 8-bit). Any
  loop-filter change must be re-verified at 10/12-bit, not just 8-bit.
- **The whole per-frame loop filter is gated on the frame-level `loop_filter_level` (spec §8.1),
  before per-block ref/mode deltas apply.** Those deltas can raise a block's level above 0, so
  running the filter on a frame that signaled "no filtering" is wrong.
- **`decode_block` rejects two malformed-bitstream conditions before the residual/predict path**
  (which indexes fixed size/tx tables and unwraps reference views without re-checking): a
  `BLOCK_INVALID` chroma block size, and an inter block referencing an empty DPB slot. Conformant
  streams never trip these, so they change no valid-input output -- don't remove them as "dead"
  checks. Regression-guarded by `tests/robustness_test.rs` (a fuzz for no-panic-on-bad-input).
- **Malformed-input rejection checks are conformance-safe but empirically tuned -- don't loosen
  them without re-running the full sweep AND `tests/invalid_vector_test.rs`.** To pass the
  official `invalid-*` gate the decoder rejects, before/without emitting garbage: an absurd frame
  size (`HeaderError::FrameSizeTooLarge`, cap `MAX_FRAME_LUMA_SAMPLES`, a DoS guard against an
  8 GB allocation abort), an inter reference whose bit-depth/subsampling differs from the current
  frame (`RefFrameFormatMismatch`), and a tile whose arithmetic decoder finishes far off its
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
- **Large unit-test modules live out-of-line** in a sibling `<module>/tests.rs` (declared with
  `#[cfg(test)] mod tests;`): `transform`, `header`, `loop_filter`, `tile`, `tile::mode_info`,
  and the crate root (`src/tests.rs`). Smaller modules keep their tests inline. Split a module's
  tests out when the test body grows large (a few hundred lines) rather than inlining a block
  that rivals the source in size.

## Known gaps

- **No SIMD for the 10/12-bit path, reference-scaled inter prediction, intra prediction, or the
  ADST / WHT / mixed inverse transforms** (the common 8-bit DCT_DCT transform *is* vectorized;
  nor aarch64 NEON). The scalar path there is correct and bit-exact; this is a performance gap
  only, tracked in `docs/backlog.md`. (Intra prediction profiled at ~0.3% on inter content.)
- **`SEG_LVL_ALT_L` / `SEG_LVL_REF_FRAME` / `SEG_LVL_SKIP` have no official test vector.** They
  are proven instead by synthetic round-trip vectors (`tests/synthetic_seg_test.rs`)
  cross-decoded byte-identically by ffmpeg's `libvpx-vp9` and native `vp9` decoders.
- **19 corpus clips ship no upstream `.md5`** (`vp90-2-bbb_*` / `vp90-2-tos_*` /
  `vp90-2-sintel_*` movies), so the sweep can't MD5-check them and they're excluded from the 315.
  Instead they're cross-checked against ffmpeg's `libvpx-vp9`: the 12 tos/sintel clips in full
  (268,832 displayed frames byte-identical), and the 7 bbb clips as a spot-check — first 1000
  frames each, 7000 total, byte-identical (`tests/bbb_cross_check.rs`).
