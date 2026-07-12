# Implementation notes

Records spec-external judgment calls, tradeoffs, and known gaps that aren't obvious
from reading the code/comments alone. Update this file as such decisions are made.

## Full segmentation support (2026-07-12)

### Scope

Implemented full segmentation (spec §6.2.11, §6.4.7, §6.4.9, §6.4.12, §6.4.14-6.4.17,
§7.2.10, §8.1 step 3, §8.6.1, §8.8.1), replacing the previous M1/M2 stub that parsed
the bitstream but discarded it and rejected any frame with `segmentation_enabled == 1`
(`TileError::SegmentationNotSupported`, now removed).

### Changes by area

- `src/header.rs`: `parse_segmentation_params()` now returns a full `SegmentationParams`
  (`enabled`, `update_map`, `tree_probs[7]`, `pred_prob[3]`, `temporal_update`,
  `abs_or_delta_update`, `feature_enabled[8][4]`, `feature_data[8][4]`) instead of a
  bare `bool`. `feature_enabled`/`feature_data`/`abs_or_delta_update` persist across
  frames (threaded through `Decoder::segmentation_features` in `lib.rs`, mirroring the
  pre-existing `loop_filter_deltas` persistence pattern) and are reset to
  zero/false under the same `FrameIsIntra || error_resilient_mode` condition as the
  loop filter deltas (`setup_past_independence()`, spec §7.2). `tree_probs`/`pred_prob`
  need no cross-frame persistence: they're read fresh whenever `update_map`/
  `temporal_update` is set and are never referenced otherwise.
- `src/tile.rs`: added `intra_segment_id()` (§6.4.7, tree-decoded via the new
  `SEGMENT_TREE` in `prob_tables.rs`), `inter_segment_id()` (§6.4.12, with
  `AboveSegPredContext`/`LeftSegPredContext` tracking for `seg_id_predicted`), and
  `get_segment_id()` (§6.4.14, min-over-block-region lookup into `PrevSegmentIds`).
  `seg_feature_active()` gates `read_skip` (SEG_LVL_SKIP), `read_is_inter` and
  `read_ref_frames` (SEG_LVL_REF_FRAME), and the top of `inter_block_mode_info`
  (SEG_LVL_SKIP forces `y_mode = ZEROMV` without reading `inter_mode`). Each gated
  site returns before touching `counts`, so backward adaptation never sees a count for
  a syntax element that wasn't actually read.
- `src/lib.rs`: `Decoder` gained `prev_segment_ids: Vec<u8>` (row-major
  `MiRows x MiCols`, unpadded) implementing `PrevSegmentIds`. Cleared to all-zero
  before tile decode (`clear_prev_segment_ids_if_needed`) under either of two spec
  conditions: `compute_image_size`'s frame-size-changed-or-first-invocation rule
  (spec §7.2.6 — reuses the existing `prev_frame_dims` comparison already used for
  `UsePrevFrameMvs`), or `setup_past_independence()` (spec §7.2, `FrameIsIntra ||
  error_resilient_mode` — matching libvpx's `vp9_setup_past_independence`, which
  memsets `last_frame_seg_map`). The clear must happen *before* `decode_tiles`
  because an error-resilient inter frame reads the map during its own decode
  (`update_map == 0`, or temporal prediction with `seg_id_predicted == 1`).
  Refreshed from this frame's decoded `SegmentIds` (read out of
  `TileDecoder::mi_grid()` after `decode_tiles`) only when
  `segmentation_enabled && segmentation_update_map` (spec §8.1 step 3 — explicitly
  *not* gated by `show_frame`, unlike most other post-decode state).
  [Fix 2026-07-12: the initial implementation only cleared on
  first-frame/size-change and missed the `setup_past_independence` reset entirely,
  leaving stale ids readable by (a) an error-resilient inter frame with
  segmentation enabled, and (b) inter frames following a same-size mid-stream
  keyframe/intra-only frame that didn't refresh the map. Covered by the
  `prev_segment_ids_reset_lifecycle` unit test in `src/lib.rs`.]
- `src/quant.rs`: `get_qindex()`/`SegQIndexOverride` were already spec-compliant
  (reviewed in an earlier milestone) and needed no logic changes — only wired up for
  real from `tile.rs` (previously the caller always passed `None`).
- `src/loop_filter.rs`: `build_lvl_lookup()` gained the SEG_LVL_ALT_L branch (spec
  §8.8.1 step 2: absolute-or-delta override of `lvlSeg` before the existing
  ref/mode-delta computation), reachable via a new `seg: &SegmentationParams`
  parameter threaded through `loop_filter_frame()`/`TileDecoder::apply_loop_filter()`.

### Judgment calls

- **`PrevSegmentIds` kept separate from `prev_mi_grid`.** `MiInfo::segment_id` in
  `prev_mi_grid` (used for MV prediction) is written for *every* block every frame,
  including `segment_id = 0` when segmentation is disabled that frame. But spec §8.1
  step 3 only refreshes `PrevSegmentIds` when `segmentation_enabled &&
  segmentation_update_map` — so if segmentation is briefly disabled for one frame and
  re-enabled the next with `update_map == 0`, the predicted ids must still come from
  the last frame that had `update_map == 1`, not from the intervening all-zero frame.
  Reusing `prev_mi_grid` directly would have silently zeroed the prediction in that
  case. Also, `prev_mi_grid`'s pass-through to `TileDecoder` is gated by
  `use_prev_frame_mvs` (which additionally depends on `error_resilient_mode`/
  `show_frame`/`FrameIsIntra` — conditions §7.2.6 attaches to MV prediction, not
  segmentation), so gating `PrevSegmentIds` the same way would have been wrong for a
  different reason too. A dedicated `Decoder::prev_segment_ids` field, with its own
  reset/refresh conditions copied verbatim from the spec text, avoided both traps.
- **`SEG_LVL_SKIP` + sub-8x8 blocks.** The spec states as a bitstream-conformance
  requirement (not a decoder obligation) that `seg_feature_active(SEG_LVL_SKIP)`
  implies `MiSize >= BLOCK_8X8` whenever `inter_block_mode_info` runs. No runtime
  check was added for this (consistent with how this codebase generally trusts
  conformant input elsewhere); the `MiSize < BLOCK_8X8` sub8x8 loop in
  `inter_block_mode_info` was left untouched since it's spec-guaranteed unreachable
  under seg-skip.
- **`SegQIndexOverride` kept as its own struct** (in `quant.rs`) rather than having
  `get_qindex` take `&SegmentationParams` + `segment_id` directly, so `quant.rs` stays
  testable/usable standalone without a `header.rs` dependency (matches its pre-existing
  design from before this change).

### Conformance-vector gap

Segmentation is unit-tested (`src/header.rs`: parse round-trip + persistence/reset;
`src/tile.rs`: `intra_segment_id`/`inter_segment_id`/`get_segment_id` and seg-feature
gating of skip/is_inter/ref_frames; `src/loop_filter.rs`: SEG_LVL_ALT_L level
derivation) but has **no end-to-end conformance vector**. Checked libvpx's official
`test/test-data.sha1` (via GitHub): the VP9 segmentation vectors
(`vp90-2-15-segkey.webm`, `vp90-2-15-segkey_adpq.webm`) exist only in `.webm` form
upstream — no `.ivf` variant is published. The `.ivf` files matching `*segmentation*`
in that same manifest (`vp80-03-segmentation-*.ivf`) are VP8, not VP9, and are not
decodable by this crate. Per the task constraints, container conversion (`.webm` ->
`.ivf`) was not performed. `tests/vectors/` therefore still contains only the two
pre-existing non-segmentation vectors, and `cargo test` passing unchanged on them
confirms no regression on the segmentation-disabled path, but does not exercise the
segmentation code paths added here against an official MD5.

## Frame-context store over-reset for intra_only frames (fixed 2026-07-12)

### What was wrong

`Decoder::decode_frame` (`src/lib.rs`) called `self.frame_contexts.reset_all()`
(wiping all 4 stored probability-table slots back to defaults) whenever
`header.frame_is_intra || header.error_resilient_mode`. Since `frame_is_intra` is
true for both key frames *and* intra_only frames, this over-reset all 4 slots for
every intra_only frame too, regardless of the bitstream's `reset_frame_context`
value. `reset_frame_context` was parsed (`src/header.rs`) but never consumed as
decoder state — a pure dead read.

### What the spec requires

Spec §7.2 `setup_past_independence()`, called only when `FrameIsIntra ||
error_resilient_mode` (confirmed against the reference decoder,
`vp9_setup_past_independence()` in libvpx's `vp9/common/vp9_entropymode.c`):

- `frame_type == KEY_FRAME || error_resilient_mode || reset_frame_context == 3`:
  reset **all 4** stored context slots to defaults.
- else if `reset_frame_context == 2`: reset **only** the slot at `frame_context_idx`
  (the raw bitstream value, read *before* `setup_past_independence()` forces
  `frame_context_idx` to 0 — see below).
- else (`reset_frame_context` 0 or 1): reset **no** stored slot; the frame's
  starting probabilities come from whatever is already retained in
  `frame_contexts[frame_context_idx]` (slot 0, since `frame_context_idx` is forced
  to 0 for every `FrameIsIntra || error_resilient_mode` frame, intra_only included).

Key/error-resilient frames were already spec-compliant before this fix, since their
correct behavior (full reset) happens to coincide with the old unconditional
`reset_all()`.

### Changes

- `src/header.rs`: added `NewFrameHeader::frame_context_idx_raw`, the raw `f(2)`
  bitstream value, alongside the existing `frame_context_idx` (which continues to
  hold the post-`setup_past_independence` forced value, unchanged, still used as-is
  for `load_probs`/`save_probs`/`refresh_frame_context`). The raw field exists
  solely because `reset_frame_context == 2` targets the pre-forcing index.
- `src/lib.rs`: added `frame_context_reset()`, a pure function mirroring the four-way
  branch above, and its `FrameContextReset::{None, Slot, All}` result. `decode_frame`
  now matches on it instead of unconditionally calling `reset_all()`.
- `src/loop_filter.rs`: corrected a module-doc comment that claimed loop filter
  `ref_deltas`/`mode_deltas` persistence across frames was unimplemented; it has in
  fact been implemented since M3 (`Decoder::loop_filter_deltas` in `lib.rs`,
  threaded into `parse_loop_filter_params` in `header.rs`).
- `README.md`: updated the M3-era changelog entry that described `reset_frame_context
  == 2` as an unimplemented known limitation.

### Judgment calls / tradeoffs

- Kept `frame_context_idx` (forced) and `frame_context_idx_raw` as two separate
  struct fields rather than repurposing the existing field, to avoid any risk of
  silently changing the value used by the (already spec-correct) `load_probs`/
  `save_probs` call sites.
- `frame_context_reset()` takes `frame_context_idx_raw` as a plain `u8` parameter
  (not tied to `NewFrameHeader`) so it's a pure, directly unit-testable function per
  the review's suggestion.

### Verification gap

`cargo test` (119 tests total, including the two MD5 full-frame conformance vectors)
passes unchanged, which confirms key-frame and error-resilient decoding is
bit-identical to before. The new reset-selection logic (`reset_frame_context` values
0/1/2 on intra_only frames) is covered only by unit tests in `src/lib.rs`
(`frame_context_reset_*`, `frame_context_store_reset_application`) that exercise the
pure decision function and `FrameContextStore` directly — there is no bitstream-level
conformance vector in `tests/vectors` that actually exercises an intra_only frame
with `reset_frame_context < 3`, so this path has not been verified against a real
encoded stream or an official MD5.
