# Implementation notes

Records spec-external judgment calls, tradeoffs, and known gaps that aren't obvious
from reading the code/comments alone. Update this file as such decisions are made.

## Current state index

Quick pointers from topic to the entry reflecting *current* behavior -- earlier entries on
the same topic may describe a since-fixed bug; read forward from here, not top-to-bottom.

- **Segmentation** (seg-id decode, all 4 `SEG_LVL_*` features): "Full segmentation support"
  (2026-07-12); coverage detail in "Conformance coverage" (2026-07-13) and "Synthetic
  round-trip coverage for SEG_LVL_ALT_L / SEG_LVL_REF_FRAME / SEG_LVL_SKIP" (2026-07-14).
- **Frame-context reset** (`reset_frame_context` handling): "Frame-context store over-reset
  for intra_only frames" (fixed 2026-07-12).
- **Superframe splitting**: "missing VP9 superframe splitting" (2026-07-13), superseded by
  "Superframe splitting moved into the public `decode_frame` API" (2026-07-14) -- the latter
  is the current public API contract.
- **Coefficient EOB-branch counting**: "Coefficient EOB-branch over-counting corrupts
  adapted coef context" (fixed 2026-07-14).
- **Synthetic round-trip + external cross-decode harness**: "Synthetic round-trip coverage
  for SEG_LVL_ALT_L / SEG_LVL_REF_FRAME / SEG_LVL_SKIP" (2026-07-14).
- **WebM remux tooling**: "WebM remux for official segmentation/intra-only vectors"
  (2026-07-13, pure-std since 2026-07-14).
- **Design-debt redesign** (current architecture -- see README.md's "Current architecture"
  section for the end state): "Wave 1" through "Wave 6" below, in date order; "Design-debt
  redesign: closing summary" at the very end ties all six together.

Append-only from here on: when a later fix or discovery corrects a claim an earlier entry
made, add a new dated entry describing the correction and cross-link the entry it
supersedes (as this index and several entries below already do) -- do not edit old entries
in place, even to fix a claim now known to be wrong.

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

## Conformance coverage (2026-07-13)

### Scope

Added a read-only observation surface (`Decoder::last_frame_info()` /
`FrameDecodeInfo`, `src/lib.rs`) so tests can assert that a decoded stream actually
exercised a given decode path (segmentation, intra_only, `reset_frame_context`
values), not just that the output MD5 matched. Wired up (skip-cleanly-if-absent)
conformance tests in `tests/conformance_test.rs` for the three official vectors that
exercise segmentation and intra_only frames, plus a coverage-summary `eprintln!` (set
of `reset_frame_context` values seen, union of `seg_features_active`) so a human
reading the test log can see exactly which paths a vector exercises. The vectors
themselves (`vp90-2-15-segkey.webm`, `vp90-2-15-segkey_adpq.webm`,
`vp90-2-16-intra-only.webm`, remuxed to `.ivf`) are not yet present in
`tests/vectors/`, so all three new tests currently skip cleanly.

### Coverage table

| Path | Official E2E vector | Synthetic round-trip vector | Unit test only |
| --- | --- | --- | --- |
| Segmentation seg-id decode (`intra_segment_id`/`inter_segment_id`/`get_segment_id`) | ✓ `vp90-2-15-segkey` -- `intra_segment_id` only (1-frame vector, no inter frame, so `inter_segment_id`/`get_segment_id` temporal prediction remain unproven E2E) | TODO | yes (`src/tile.rs`) |
| `SEG_LVL_ALT_Q` | ✓ **`vp90-2-15-segkey_adpq`** -- all 150 output frames bit-exact (coverage line reports `SEG_LVL_ALT_Q=true`) | TODO | yes (`src/quant.rs`, `src/header.rs`) |
| `SEG_LVL_ALT_L` | none (no official or readily-encodable vector) | ✓ **`tests/synthetic_seg_test.rs`** (2026-07-14; ffmpeg cross-decoded -- see the External cross-decode sections below) | yes (`src/loop_filter.rs`) |
| `SEG_LVL_REF_FRAME` | none (no official or readily-encodable vector) | ✓ **`tests/synthetic_seg_test.rs`** (2026-07-14; ffmpeg cross-decoded -- see the External cross-decode sections below) | yes (`src/tile.rs`) |
| `SEG_LVL_SKIP` | none (no official or readily-encodable vector) | ✓ **`tests/synthetic_seg_test.rs`** (2026-07-14; ffmpeg cross-decoded -- see the External cross-decode sections below) | yes (`src/tile.rs`) |
| `intra_only` frame | ✓ **`vp90-2-16-intra-only`** -- all 7 output frames bit-exact; the 3 hidden `intra_only` priming frames are verified via `show_existing_frame` round-trip and the 4 inter frames referencing them decode correctly (coef EOB-count fix below) | TODO | yes (`src/header.rs`) |
| `reset_frame_context < 3` | ✓ **`vp90-2-16-intra-only`** -- exercises `reset_frame_context == 2` on all 3 of its `intra_only` frames (coverage line reports `reset_frame_context values seen = {0, 2}`) | TODO | yes (`src/lib.rs`: `frame_context_reset_*`) |

Final state (as of the superframe-splitting-API and EOB-count fixes below): seg-id decode,
`SEG_LVL_ALT_Q`, `intra_only`, and `reset_frame_context < 3` (rfc==2 seen) all have official
end-to-end coverage. `SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP` have no official vector
(still true as of 2026-07-14 -- libvpx ships none, see the entry below), but as of 2026-07-14
have a synthetic round-trip vector instead (`tests/synthetic_seg_test.rs`): this is a
self-consistent encoder-in-this-repo <-> decoder-in-this-repo check, not an official-MD5
conformance pass, so the table still distinguishes it from the "Official E2E vector" column.

## Synthetic round-trip coverage for SEG_LVL_ALT_L / SEG_LVL_REF_FRAME / SEG_LVL_SKIP (2026-07-14)

### Scope

The "Conformance coverage" entry above left `SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP`
unit-test-only, since libvpx ships no official (or readily re-encodable) IVF vector exercising
them (`vp90-2-15-segkey*.webm` only exercise `SEG_LVL_ALT_Q`/plain seg-id decode; libvpx's
`vp80-03-segmentation-*.ivf` vectors are VP8, not VP9). Rather than leave the gap, added
`tests/synthetic_seg_test.rs`: a self-contained, pure-std (no dependencies of any kind --
written under the then-current policy that forbade even dev-dependencies, and still true of this
file after the 2026-07-15 relaxation for test tooling) synthetic VP9 *encoder*, hand-rolled
inside the test file
itself, that emits just enough bitstream to drive each of the three features through
`Decoder::decode_frame` and checks the result.

### What this is (and isn't)

This is a **round-trip** test: this file's own bool-coder/bit-writer encoder feeds bytes to
this crate's own `Decoder`. It is not conformance against an official MD5 -- within the round trip alone there is no
independent reference encoder or decoder, so a bug present identically on both the encoding
assumptions here and the decoder's implementation would not be caught by it (the "External
cross-decode result" section below closes exactly this blind spot with two independent
third-party decoders). What it
does prove is that the decoder actually takes the three `seg_feature_active`-gated forced code
paths in `src/tile.rs` (`read_skip`, `read_is_inter`, `read_ref_frames`, and
`inter_block_mode_info`'s `SEG_LVL_SKIP` ⇒ `y_mode = ZEROMV` branch) and the `SEG_LVL_ALT_L`
branch of `src/loop_filter.rs::build_lvl_lookup`, instead of silently falling back to reading
ordinary per-block bits or ignoring the segmentation override: the hand-built bitstreams contain
none of the bits a fallback path would need, so a bypassed feature either desyncs the decode
outright or reconstructs pixels that don't match the hand-derived expected values asserted in
the tests. This was verified directly, not just argued: temporarily flipping each
`feature_enabled` flag off one at a time (while leaving the bitstream otherwise unchanged) was
confirmed to make the corresponding test fail -- for `SEG_LVL_REF_FRAME` and `SEG_LVL_SKIP` via
an actual pixel divergence (the inter frame decoded as flat grey instead of reproducing the key
frame), for `SEG_LVL_ALT_L` via the two decodes (levels 0 and 63) becoming identical instead of
differing. (These were manual, throwaway edits made and reverted during development, not
committed as part of the test file.)

### Why no residual-token encoder

Every block in every synthetic frame uses `skip = 1`, so no coefficient/token bits are ever
encoded or decoded. Writing a residual-token encoder (inverse of `src/tile.rs::tokens_and_reconstruct`)
would have been substantial additional pure-std code for no benefit here: with `skip = 1`, the
only pixel values that can appear are the direct output of intra prediction
(`src/predict.rs::predict_intra`) or, for the inter-frame test, zero-MV motion compensation
(an exact copy of the reference) -- both of which are simple enough to hand-verify pixel-by-pixel
without needing residuals to make the image non-flat. The "flat content" limitation this implies
(a single isolated intra block, or a block whose neighbors are all real/available, tends to
predict flat or to converge back to its neighbor's value -- see the `SEG_LVL_SKIP`/`SEG_LVL_REF_FRAME`
test's code comment for a worked example of a 4-different-y_modes layout collapsing to a single
flat 127 value once causal neighbor propagation is taken into account) was worked around by
choosing modes that are *structurally* immune to neighbor propagation in the direction that
matters: `V_PRED` (which only ever reads the row above, and is given no "above" neighbor at all
since it's used only in row 0) and `H_PRED` (which never reads "above" at all, so it can't
inherit row 0's value even though row 1 does have a real "above" neighbor).

### Test-by-test summary

- **`seg_lvl_skip_forces_exact_copy_without_residual_or_mv_bits`** and
  **`seg_lvl_ref_frame_selects_the_forced_reference_without_reading_bits`** share one scenario
  (`decode_skip_ref_frame_scenario`): a key frame (V_PRED row 0 / H_PRED row 1, `skip = 1`,
  giving an exact, hand-verified 127/129 split -- not a single flat value) followed by an inter
  frame with one segment forcing `SEG_LVL_SKIP` and `SEG_LVL_REF_FRAME = LAST_FRAME` for every
  block. Proven: `seg_features_active[SEG_LVL_SKIP]`/`[SEG_LVL_REF_FRAME]` are both set, and the
  inter frame's Y/U/V are bit-identical to the key frame's (ZEROMV + forced LAST + forced skip
  ⇒ an exact copy). Left open here (every DPB slot holds the same key frame, since key frames
  refresh all 8 slots, so there is only one candidate reference to copy from either way): that
  `SEG_LVL_REF_FRAME` steers to a *specific* slot, as opposed to always falling through to
  whatever the single reference happens to be. Closed by the next test.
- **`seg_lvl_ref_frame_steers_to_the_specific_slot_not_just_last`** (2026-07-14, added after the
  above): closes that gap for the GOLDEN case. Three frames: (1) a key frame, content A = flat
  127, refreshing all 8 DPB slots; (2) a new hidden (`show_frame = 0`) `intra_only` frame, content
  B = flat 129, refreshing *only* physical slot 1 (`refresh_frame_flags = 0x02`), leaving slot 0
  at A -- built with the new `build_intra_only_header` helper (its tail is bit-for-bit identical
  to `build_keyframe_header`'s, since both share `frame_is_intra = true`, verified against
  `src/header.rs::parse_uncompressed_header`'s intra_only branch and the shared post-frame-size
  tail that follows it; its tile data and compressed header reuse `encode_keyframe_tile`/
  `build_keyframe_compressed_header` unchanged, since `frame_is_intra` also makes an intra_only
  frame's tile/compressed-header shape identical to a key frame's); (3) an inter frame with
  `ref_frame_idx = [0, 1, 2]` (LAST -> slot 0 = A, GOLDEN -> slot 1 = B) and
  `SEG_LVL_REF_FRAME`'s `feature_data = GOLDEN_FRAME`. Discriminating assertion: every pixel of
  the inter frame's Y plane is 129 (GOLDEN's/slot 1's content) -- a decoder that read
  `FeatureData` but resolved the wrong slot (even for only some blocks), or one that ignored it
  and always used LAST, would leave 127 somewhere. A same-decoder companion decode steered to
  LAST is additionally asserted to be uniformly 127, pinning in-test that slot 0 still held A --
  so the GOLDEN=129 check can't be satisfied by an over-refresh that put B in every slot.
  **ALTREF was not additionally exercised**
  (judged low marginal value once GOLDEN vs. LAST is proven via the same `resolved_refs[..]`
  indexing code path; left as a possible future addition, not required by this task). Like the
  rest of this file, still a self-consistent round trip against this crate's own decoder, not
  conformance against an official MD5.
- **`seg_lvl_alt_l_loop_filter_level_change_is_observable`**: a key frame with 2 segments (row 0
  segment 0, no `ALT_L`; row 1 segment 1, `ALT_L` under test, absolute override), decoded twice
  with segment 1's `ALT_L` level at 0 and 63. Oracle: exact hand-derived pixel values, not just
  "the outputs differ" -- with `narrow_filter`'s formula worked out by hand for the 127/129 edge
  (see the test's code comment), level 63 pulls all four samples touching the edge (rows 6-9) to
  exactly 128, while level 0 leaves them untouched (127/127/129/129); rows 5 and 10, one step
  further from the edge, are asserted unchanged at either level as an anchor that only the edge
  moved. This is the strong form of the oracle the task asked for (stronger than the "assert
  outputs differ" fallback, and stronger than the "reproduces the seg-disabled baseline"
  equivalence check, which was not additionally implemented given the exact-value check already
  subsumes it as a correctness signal).

### Verification

`cargo test --test synthetic_seg_test -- --nocapture`: 6/6 pass (4 substantive tests plus the
env-gated `dump_synthetic_ivf_for_external_cross_decode` no-op and the ffmpeg-gated
`synthetic_streams_cross_decode_against_ffmpeg`, each of which skips cleanly when its external
prerequisite is absent). Full `cargo test`: 146/146 pass (was 143/143 before this work; no
regressions; see the README/task-runner output for the exact per-file breakdown).
`cargo clippy --tests`: no warnings from `tests/synthetic_seg_test.rs` (the warnings it reports
are pre-existing, in `src/`, and unrelated to this file). `cargo fmt` (checked with
`rustfmt --check` scoped to the new file only, since `examples/*.rs` and some other `tests/*.rs`
files have pre-existing formatting drift unrelated to this change) is clean.

No `src/` changes were made -- `Decoder::last_frame_info()`'s existing `seg_features_active`
observation surface (added in the "Conformance coverage" entry above) was sufficient.

### External cross-decode dump harness

The three scenarios above are now also exposed as builder fns (`build_skip_ref_frames`,
`build_steering_frames`, `build_alt_l_frames`) returning the exact ordered raw VP9 frame bytes;
the existing tests were refactored to call these fns instead of duplicating the bitstream
construction, so what gets dumped is guaranteed identical to what the tests decode. An env-gated
test, `dump_synthetic_ivf_for_external_cross_decode` (a no-op unless `VP9DEC_DUMP_DIR` is set, so
a plain `cargo test` run stays green), writes each scenario to that directory as an `.ivf`
(via a small pure-std `ivf_wrap` writer mirroring `src/ivf.rs`'s reader layout) plus this
decoder's own decoded output as raw I420 `.yuv` (shown frames only, in display order):

    VP9DEC_DUMP_DIR=<dir> cargo test --test synthetic_seg_test dump_synthetic_ivf_for_external_cross_decode -- --nocapture

(The `VAR=value` command prefix is bash-only; on PowerShell use
`$env:VP9DEC_DUMP_DIR = "<dir>"; cargo test ...`.)

This only emits the streams for an external VP9 decoder (e.g. ffmpeg) to independently decode and
compare against `*.our_i420.yuv`; the dump harness itself does not invoke ffmpeg (see below for the
separate test that does). The result of actually running that comparison is recorded next.

### External cross-decode result (2026-07-14)

The dump harness was run and each `.ivf` decoded by ffmpeg N-121910 (2025-11), a full
libvpx-enabled build, with BOTH its `libvpx-vp9` decoder (the reference implementation) and its
native `vp9` decoder (a fully independent implementation), output as raw I420 (`-f rawvideo
-pix_fmt yuv420p`) and byte-compared against this decoder's `*.our_i420.yuv`:

- `skip_ref` (`SEG_LVL_SKIP` + `SEG_LVL_REF_FRAME = LAST`): **byte-identical** under both decoders
  (768 B, 2 shown frames).
- `alt_l_0` / `alt_l_63` (`SEG_LVL_ALT_L` absolute level 0 vs. 63): **byte-identical** under both
  decoders (384 B each). This independently confirms the hand-derived loop-filter output (the
  127/129 edge collapsing to a flat 128 at level 63) matches libvpx and ffmpeg-native exactly --
  i.e. `build_lvl_lookup`'s `SEG_LVL_ALT_L` branch is correct, not merely internally
  self-consistent.
- `steering` (`SEG_LVL_REF_FRAME = GOLDEN` slot-steering): the **shown** frames are byte-identical
  under both decoders -- our key frame (127) equals ffmpeg's first output frame, and our inter
  frame (129) equals ffmpeg's last output frame. ffmpeg emits one *extra* frame here: it also
  outputs the hidden (`show_frame == 0`) `intra_only` frame (content 129), which this decoder
  deliberately does *not* display (spec §8.9 output process: only `show_frame == 1` /
  `show_existing_frame` frames are output; the official `.ivf.md5` files likewise carry no line
  for hidden frames). So the size mismatch (ffmpeg 1152 B / 3 frames vs. ours 768 B / 2 frames) is
  a frame-*output*-policy difference, not a pixel discrepancy -- frame-for-frame the pixels agree,
  and ffmpeg's copy of the hidden frame (129) additionally confirms this decoder's internal decode
  of it.

This obtains the substantive thing official-MD5 conformance provides -- an *independent* oracle --
despite no official vector existing for these three features: all three
`SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP` code paths now produce output confirmed
byte-identical by two independent third-party VP9 decoders, closing the round-trip's one
theoretical blind spot (a bug shared identically by this repo's encoder assumptions and its
decoder). It is still *not* "official MD5 conformance" in the literal sense (the streams are our
own synthetic ones, not libvpx-published vectors with published MD5s), and ffmpeg remains a
reference/debugging tool only -- not a build or crate dependency (the automated test below invokes
the binary out-of-process but skips cleanly without it).

This comparison is no longer a one-time manual check: `synthetic_streams_cross_decode_against_ffmpeg`
(in the same test file) automates it. It reuses the same scenario builders/`ivf_wrap` as the dump
harness above -- via a shared `scenarios()` list, so the dumped set and the cross-checked set
cannot diverge -- and shells out to the ffmpeg *binary* via `std::process::Command` (located via
the `VP9DEC_FFMPEG` env var, falling back to `ffmpeg` on `PATH` -- never a hardcoded path; a
set-but-unusable `VP9DEC_FFMPEG` fails the test rather than silently skipping). It runs whichever
of `libvpx-vp9` and `vp9` (ffmpeg's own independent native decoder) the build provides per
`-decoders` -- both on a full build; a build lacking one is noted and cross-decoded with the other
alone, one lacking both skips. Each run requires a clean ffmpeg stderr and an output of exactly
one frame per constituent VP9 frame (hidden frames included), then byte-compares every shown
frame against its constituent index; the exact-length requirement is load-bearing, since
steering's hidden frame and its inter frame are pixel-identical and a laxer length check could
alias a dropped final frame onto the hidden one. If no ffmpeg binary is found (probed via
`ffmpeg -version`), the test prints a `[skip]` line and passes trivially, so a plain `cargo test`
on a machine without ffmpeg installed stays green -- same skip-if-absent convention as the
vector-file-gated conformance tests. This still adds zero crate dependencies (only the ffmpeg
*binary* is invoked, never linked), so the product and test harness remain pure-std. The
shown-frame byte-identical result recorded above is therefore re-checked on every run where an
ffmpeg binary is available, no longer a single point-in-time observation; the steering scenario's
hidden-frame confirmation, by contrast, remains manual-only (the public decoder API never
surfaces hidden frames' pixels for the automated test to compare).

## WebM remux for official segmentation/intra-only vectors (2026-07-13, pure-std since 2026-07-14)

### Scope

Added `examples/webm_to_ivf.rs`, a WebM -> IVF remuxer, so the three official libvpx VP9
vectors that upstream ships only as `.webm` (`vp90-2-15-segkey`, `vp90-2-15-segkey_adpq`,
`vp90-2-16-intra-only`) can run through the existing `IvfReader`-based
`tests/conformance_test.rs` harness. Container change only (no re-encode): each WebM
(Simple)Block's frame payload is copied byte-for-byte into an IVF frame record.

### Std-only, no dependencies (revised 2026-07-14)

The remuxer is written against the Rust standard library alone, with no crate
dependency of any kind. An initial version used the `matroska-demuxer` crate as a
`[dev-dependencies]` entry; that was replaced because this repo's zero-dependency policy as then
worded in the README ("zero dependencies, including dev-dependencies ... only the Rust standard
library") extended to dev-dependencies too. (On 2026-07-15 that policy was relaxed to allow
dev-dependencies for test tooling -- the shipped decoder stays zero-dependency -- but this
example remains pure-std.) ffmpeg/ffmpeg-sys bindings were never an option for the shipped
decoder for the same reason (and would pull in a C toolchain besides).

The replacement is a small hand-rolled EBML/Matroska reader (~330 lines, all in the
example file, never referenced from `src/`):

- **vints**: element IDs are read keeping their length-marker bits (stored verbatim per
  the EBML spec); size/value vints strip the marker bit. The all-ones size vint is
  decoded as "unknown size". The first-byte value mask is computed as `0xFF >> length`
  with an explicit `length >= 8 => 0` guard (an 8-byte vint's first byte is pure
  length-descriptor, and `0xFFu8 >> 8` would panic).
- **descent**: top level -> skip the EBML header, find `Segment`; within Segment ->
  `Tracks` (walk `TrackEntry`s for the one with `TrackType == 1` and `CodecID ==
  "V_VP9"`, reading its `TrackNumber` and `Video`/`PixelWidth`/`PixelHeight`), then each
  `Cluster` -> `SimpleBlock` (and `BlockGroup`/`Block` if present). Unknown element IDs
  are skipped by their declared size; unknown *structure* (an unexpected unknown-sized
  element, an overrunning size, a truncated block) is a hard error rather than a guess.
- **unknown-sized masters**: a `Cluster` with unknown size is bounded by scanning forward
  to the next Segment-level (level-1) element ID (`SEGMENT_LEVEL_IDS`), matching how the
  EBML spec says an unknown-sized master ends. (These particular vectors happen to use
  definite sizes throughout, but the handling is present and correct so the example
  isn't silently file-specific.)

### Lacing guard

Each block's flags byte is read directly and the lacing bits (`(flags >> 1) & 0x03`)
must be 0; a non-zero value (Xiph/EBML/fixed-size lacing) is a hard error rather than a
guess. VP9-in-WebM is one frame per block, and a VP9 superframe is an opaque
payload-internal concept that passes through as verbatim bytes (superframe *splitting*
happens later, inside the decoder -- see the superframe sections below). All three
vectors have exactly one `V_VP9` track and no laced blocks, so the guard never fires.

### Verification (byte-identical to the crate version, then 7/7)

The pure-std remux was validated by regenerating all three `.ivf` files and byte-comparing
(`cmp`) them against the ones the earlier `matroska-demuxer` version produced: **all three
are byte-identical**. So the elementary-stream payloads fed to the decoder are provably
unchanged by the dependency removal. Frame counts also match each `.ivf.md5`'s line count
(segkey: 1, segkey_adpq: 150, intra-only: 7). With the decoder fixes recorded in the
sections below (superframe splitting + coefficient EOB-count), all three conformance
tests now pass; `cargo test --test conformance_test -- --nocapture` reports:

```text
[ok] .../vp90-2-15-segkey.ivf: all 1 output frames exactly match the official MD5
[coverage] .../vp90-2-15-segkey.ivf: reset_frame_context values seen = {0}; seg_features_active union: SEG_LVL_ALT_Q=false SEG_LVL_ALT_L=false SEG_LVL_REF_FRAME=false SEG_LVL_SKIP=false
[ok] .../vp90-2-15-segkey_adpq.ivf: all 150 output frames exactly match the official MD5
[coverage] .../vp90-2-15-segkey_adpq.ivf: reset_frame_context values seen = {0}; seg_features_active union: SEG_LVL_ALT_Q=true SEG_LVL_ALT_L=false SEG_LVL_REF_FRAME=false SEG_LVL_SKIP=false
[ok] .../vp90-2-16-intra-only.ivf: all 7 output frames exactly match the official MD5
[coverage] .../vp90-2-16-intra-only.ivf: reset_frame_context values seen = {0, 2}; seg_features_active union: SEG_LVL_ALT_Q=false SEG_LVL_ALT_L=false SEG_LVL_REF_FRAME=false SEG_LVL_SKIP=false
```

### Judgment calls

- `seg_features_active[level]` (in `FrameDecodeInfo`) is gated by
  `segmentation_enabled` for the current frame, not just whether `FeatureEnabled` is
  set somewhere in the persisted state — `FeatureEnabled`/`FeatureData` persist across
  frames per spec §7.2.10 even while segmentation is temporarily disabled, and a "this
  frame exercised SEG_LVL_X" signal that stayed true through such a gap would be
  misleading for coverage purposes.
- The `SEG_LVL_*` indices used in `FrameDecodeInfo::seg_features_active` are the
  existing `header::SEG_LVL_ALT_Q`/`SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP`
  constants (`src/header.rs`), not a new numbering — `FrameDecodeInfo` just borrows
  `header::SEG_LVL_MAX` for the array length so the two stay in sync automatically.
- `last_frame_info()` is not updated on `show_existing_frame` frames, since that path
  parses no new uncompressed header to read the values from (it re-displays an
  existing DPB slot). This matches the existing precedent of `last_frame_type` etc.
  also being frozen across such frames.

## `vp90-2-16-intra-only`: missing VP9 superframe splitting (2026-07-13)

### Correction to the prior triage

The "WebM remux" entry above guessed the panic was caused by "an over-eager check
that a slot referenced by `ref_frame_idx` has ever been written". That guess was
wrong. The actual root cause: **this decoder has no VP9 "superframe" support at
all**, and IVF frame 0 of this vector is a superframe.

Dumping the raw bytes of IVF frame 0 (117963 bytes) shows its last byte is `0xd3`
(`0b110_10_011`) -- exactly the VP9 superframe-index marker (top 3 bits `0b110`,
spec "VP9 Bitstream - superframe and uncompressed header" §3: fetched from
`https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream_superframe-and-uncompressed-header_v1.0.pdf`).
Decoding the index: `bytes_per_framesize = 3`, `num_frames = 4`, sizes
`[30299, 35391, 40875, 11384]`, summing exactly to `117963 - 14` (the index is 14
bytes). So this one IVF packet is actually **four** concatenated VP9 frames: three
hidden `intra_only` frames (refreshing DPB slots 0, 1, 2 respectively) followed by
one visible inter frame referencing `ref_frame_idx = [0, 1, 2]`
(`LAST`/`GOLDEN`/`ALTREF`). `Decoder::decode_frame` parses only the *first*
frame's uncompressed header and treats everything after it as that one frame's
own tile data, so frames 2-4 of the superframe were never decoded at all --
hence slots 1 and 2 being "never written" when the (mis-parsed) visible frame
tried to reference them. Also relevant, found via the same byte dump: IVF frames
2/3/4 (each exactly 1 byte, `0x8a`/`0x89`/`0x88`) are ordinary
`show_existing_frame` frames, not a remux artifact either.

### Fix

Added `src/superframe.rs` (`pub fn split_superframe(data: &[u8]) -> Vec<&[u8]>`),
implementing the spec's superframe-index parsing (marker check, `index[0] ==
last_byte` duplicate-marker validation, size-sum-must-equal-payload-length
validation) with a safe fallback to `vec![data]` (treat as a single ordinary
frame) whenever any of those don't hold -- this exactly matches the spec's own
description of how a decoder tells a real superframe index apart from a coded
frame that merely happens to end in a `0b110xxxxx`-shaped byte (the spec notes
encoders must avoid producing this by construction, so no valid single frame
ever has this trailing marker by accident).

`tests/conformance_test.rs`'s `check_all_frames_with_coverage` (the only helper
that runs `vp90-2-16-intra-only`) now calls `split_superframe()` on each IVF
packet and feeds every resulting VP9 frame to `decoder.decode_frame()`
individually, in order, using the same `Decoder` (so DPB/frame-context state
correctly carries across the four frames packed into IVF frame 0). This is a
container/framing-layer fix, not a change to `Decoder::decode_frame`'s contract
(it still decodes exactly one VP9 frame per call, as documented) -- deliberately
scoped to the one call site that was actually broken, per the "surgical changes"
guideline; `check_vector`/`check_all_frames` (used by the two already-passing
vectors, neither of which contains a superframe) were left untouched.

### Verification

The panic is gone. Bit-exact verification of the fix's correctness (independent
of the official `.ivf.md5`, which only covers *displayed* frames): the three
hidden `intra_only` frames inside the superframe are never displayed directly,
but IVF frames 2/3/4 (`show_existing_frame`, pointing at slots 2/1/0
respectively) *are* official MD5-checked output frames, and all three now pass --
proving the three hidden frames were decoded bit-exactly correctly, and proving
`Dpb`/`resolved_refs` resolution (the code the original panic pointed at) was
never actually at fault.

## Coefficient EOB-branch over-counting corrupts adapted coef context (fixed 2026-07-14)

### Context

After the superframe fix above, `vp90-2-16-intra-only`'s three hidden `intra_only`
frames and their three `show_existing_frame` re-displays were bit-exact, but every
frame decoded as an actual inter frame (output frames 0, 1, 5, 6) diverged frame-wide.
A validated ground-truth decode became available (ffmpeg 8.1.2 `libvpx-vp9`, verified
against the official `.ivf.md5`; its native VP9 decoder produces byte-identical output),
enabling a pixel-level diff.

### Diagnosis (diff-driven, not guesswork)

Per-frame I420 diff against ground truth: output frame 0 wrong from pixel (0,0) onward,
~92% of bytes differ, Y-plane correlation 0.53 with **matched mean/stddev** — the value
distribution is preserved but spatially scrambled, the signature of an arithmetic-decoder
desync partway through, not a motion-compensation error (which would be localized to
GOLDEN/ALTREF regions). Per-block dump showed block (0,0) is an *intra* DC-pred block
(deterministic flat-128 prediction, no reference involved) yet its residual was already
slightly wrong and block (0,2) onward catastrophic. Since the inverse transform and
intra-frame coefficient decode are proven bit-exact by the passing intra frames, and the
loaded coefficient probabilities were verified correct (coef-context fingerprints confirmed
the inter frame loads exactly what the third `intra_only` frame saved; `merge_prob`/
`adapt_coef_probs` constants match libvpx `COEF_*` = 24/112/128), the only remaining
possibility was that the *adapted* coefficient context loaded by the inter frame was
computed from wrong **counts**.

### Root cause

`TileDecoder::tokens_and_reconstruct` (`src/tile.rs`) accumulated the `more_coefs`
(EOB-branch) count incorrectly. libvpx `decode_coefs` (`vp9/decoder/vp9_detokenize.c`)
increments `eob_branch_count[band][ctx]` **once per outer-loop iteration** — i.e. only at
positions where the EOB flag is actually read: the first coefficient and every position
immediately after a non-zero token. Positions following a *zero* token are consumed in an
inner zero-run loop that reads no EOB flag and does not touch `eob_branch_count`.

This decoder instead had a "special case" `else` branch that incremented
`more_coefs[...][band][ctx][1]` at every `checkEob == 0` (post-zero) position "as if the
value were 1". That over-counts the EOB continue-branch, biasing the adapted EOB
probability (`probs[0]`, driven by `branch_ct[0] = {neob, eob_branch - neob}` in spec
§8.4.3 / libvpx `adapt_coef_probs`) toward "more coefficients". The bias lands in specific
`(band, ctx)` bins that only occur at post-zero positions.

The bug was invisible on the previously-passing vectors: `vp90-2-12-droppable_1` and
`vp90-2-09-subpixel-00` never load an intra-frame-adapted coefficient context into a
subsequent inter frame in a way that exercises the corrupted bins (their inter frames'
own decode is unaffected — counts only feed the *next* frame's adaptation). This vector is
the first where an inter frame decodes using a coefficient context that a preceding
`intra_only` frame adapted, so the corrupted EOB probabilities desync its token decode
frame-wide. The prior comment's "verified empirically via full-frame MD5 conformance"
claim was therefore only ever confirming that the extra increment *doesn't hurt* those two
vectors, never that it was correct.

### Fix

Removed the `else` branch entirely (`src/tile.rs`): `more_coefs` is now incremented only
inside the `if check_eob` block, matching libvpx's `eob_branch_count` exactly. The token
count (ZERO/ONE/TWO+) that follows is unchanged and still runs for every decoded position.
One-branch deletion, no other logic touched.

### Verification

`vp90-2-16-intra-only`: all 7 output frames now bit-exact against the official MD5
(confirmed both via the conformance test and a direct byte-compare of the decoder's I420
dump against the ffmpeg/libvpx reference `intra.yuv`). The same fix independently corrected
`vp90-2-15-segkey_adpq` (previously the separately-tracked failure at output frame 1 — it
is a 150-frame inter sequence that hits the same adapted-context path); it now passes all
150 frames. No regression: full `cargo test` green (see totals below). The
`[coverage]` line for intra-only reads:

```text
[coverage] .../vp90-2-16-intra-only.ivf: reset_frame_context values seen = {0, 2}; seg_features_active union: SEG_LVL_ALT_Q=false SEG_LVL_ALT_L=false SEG_LVL_REF_FRAME=false SEG_LVL_SKIP=false
```

`reset_frame_context` = 2 (the three `intra_only` frames' partial context reset) and 0
(the inter frames) are both exercised; `intra_only` coverage is now an official
end-to-end pass.

## Superframe splitting moved into the public `decode_frame` API (2026-07-14)

### What was wrong

The entry above ("`vp90-2-16-intra-only`: missing VP9 superframe splitting") fixed the
panic by calling `split_superframe()` only at the one call site that exercised it
(`tests/conformance_test.rs`'s `check_all_frames_with_coverage`), explicitly leaving
`Decoder::decode_frame`'s contract as "decodes exactly one VP9 frame per call". That
was fine for getting the conformance suite green, but it left the *public* API
mis-decoding on real-world input: a superframe (one or more hidden altref/intra-only
frames followed by one visible frame) is extremely common in real VP9 streams, and any
caller outside this repo's own test harness feeding `decode_frame` a raw container
chunk containing one would get garbage (the tail bytes of later sub-frames plus the
superframe index would be mis-parsed as the first sub-frame's own tile data).

### Fix

`Decoder::decode_frame(&mut self, chunk: &[u8]) -> Result<Option<Frame>, DecodeError>`
now treats its argument as one container chunk: it runs `superframe::split_superframe()`
internally and decodes each resulting VP9 frame in turn through a new private
`Decoder::decode_one_frame` (the old per-frame body, renamed/unchanged otherwise). It
returns the one displayable frame, if any (VP9 guarantees at most one visible frame per
superframe), and `last_frame_info()` reflects the *last* constituent frame decoded, not
necessarily the displayed one. `show_existing_frame` chunks (which never carry a
superframe index) pass through `split_superframe` as a single-element split, unchanged.
Every existing call site (`decode_keyframe`, `examples/decode_to_png.rs`) already passed
one container chunk (one IVF frame) per call, so no call site needed to change its
calling convention -- they were simply relying on those particular chunks never
containing more than one VP9 frame, which is no longer a requirement.

### Judgment call: kept per-subframe splitting in the coverage test

`tests/conformance_test.rs`'s `check_all_frames_with_coverage` still calls
`split_superframe()` itself and feeds each piece to `decode_frame()` individually,
rather than simplifying to one `decode_frame()` call per IVF packet. This is no longer
necessary for *decode correctness* (the whole-chunk call now produces bit-identical
output), but it's still necessary for the test's *coverage instrumentation*:
`vp90-2-16-intra-only`'s only `intra_only == true` frames are the three hidden
sub-frames of its first superframe, while the trailing visible sub-frame is an ordinary
inter frame. Since `last_frame_info()` by design reflects only the last constituent
frame of a `decode_frame()` call, sampling it once per IVF packet (instead of once per
constituent frame) would silently lose the only evidence in the whole vector that
`intra_only` was ever exercised, and the `vp90_2_16_intra_only_exercises_intra_only_frame`
coverage assertion would fail. Verified empirically: decoding is unaffected either way
(calling `decode_frame()` on an already-split sub-frame is a harmless no-op re-split),
only the granularity of what gets recorded into `last_frame_info()` differs. The
`check_all_frames`/`check_vector` helpers (used by the two vectors with no superframes)
were left as direct single `decode_frame()`-per-packet calls, since they never needed
splitting either way.

### Verification

`cargo test` green; `cargo test --test conformance_test -- --nocapture` output
byte-for-byte identical to before this change (all `[coverage]`/`[ok]` lines, 7/7
tests), confirming the refactor is decode-output-neutral.
`examples/decode_to_png.rs -- vp90-2-16-intra-only 0` (which feeds IVF frame 0 -- a real
4-frame superframe -- directly to `decode_frame()` with no pre-splitting) now decodes
and dumps correctly, which it structurally could not have before this change (the
correctness fix this section describes is specifically what makes that possible).

## Wave 1 design-debt cleanup: stale comments, dead constants, shared test infra (2026-07-16)

### Scope

Infra/trivia wave: fixed four comments that had gone stale as the decoder progressed past
the milestone they described, deleted two dead `pub` constants and one dead `pub` alias,
and de-duplicated ~120 lines of hand-copied test-only encoder/writer code (a `BoolEncoder`,
a `BitWriter`, and an IVF file writer) that had accumulated three near-identical copies
across `src/` and `tests/` because those copies had no way to see each other's
`#[cfg(test)]`-gated originals. No behavior change to the decoder itself.

### Stale comments (verified against the current code before editing)

- `src/tile.rs` module doc: no longer claims the loop filter is unimplemented --
  `TileDecoder::apply_loop_filter` has existed in the same file since M2b.
- `src/header.rs` (inter-frame `color_config` placeholder): reworded to state the actual
  contract (the parser fabricates a placeholder; `Decoder::last_color_config`, added at
  M3, overwrites it) instead of claiming the decoder still can't carry color config
  across frames.
- `src/superframe.rs` module doc: updated to match the 2026-07-14 change (see the section
  above) that moved `split_superframe` splitting inside `Decoder::decode_frame` -- callers
  no longer need to split before calling it.
- `tests/compressed_header_test.rs`: the `decode_tiles` call's inline comment claimed
  failure was "expected until token decoding is implemented" (done since M2); reworded to
  match the module doc's own (already-accurate) framing that this is a panic-only smoke
  test, full correctness being checked elsewhere.

### Dead code removed

`COLS_PARTITION_TREE`/`ROWS_PARTITION_TREE` (`src/prob_tables.rs`) and `MI_SIZE_PX` --
re-verified zero uses repo-wide (`Grep`, not just within `src/`) immediately before
deleting each, per the wave's instructions. `read_partition` (`src/tile.rs`) hand-inlines
the two 2-node partition trees rather than using the constants, so they were dead from
the day they were added.

### `test_support` module (kills the BoolEncoder/BitWriter duplication)

Moved (not copied) `BoolEncoder` out of `src/bool_coder.rs`'s `#[cfg(test)] pub(crate) mod
test_support` and `BitWriter` out of `src/header.rs`'s `#[cfg(test)] mod tests`, into a
new top-level `src/test_support.rs`, gated `#[cfg(any(test, feature = "test-support"))]`
and exposed as `pub` (previously `pub(crate)`/private, since nothing outside the crate --
nor, for `BitWriter`, outside `header.rs` -- could see them). The `test-support` feature
is enabled for integration tests (`tests/*.rs`) via a self-referencing
`[dev-dependencies] vp9dec = { path = ".", features = ["test-support"] }` entry in
`Cargo.toml`; this is not an external crate (still zero `src/` dependencies) and doesn't
affect a normal `cargo build`, confirmed by a plain `cargo build` (no features) succeeding
without compiling `test_support` at all.

`header.rs`'s other test-only helpers (`build_minimal_keyframe_header` etc.) stay in
`header.rs` as instructed -- only the generic bit-writer moved, not the header-specific
builders on top of it.

`tests/synthetic_seg_test.rs` had its own third, hand-rolled copy of both (its module doc
used to explain this was because the `src/` originals were `#[cfg(test)] pub(crate)` and
therefore invisible from `tests/`); that copy is now deleted in favor of
`use vp9dec::test_support::{BitWriter, BoolEncoder};`, and the module doc updated to
explain the `test-support` feature instead.

#### Judgment call: dropped a bit-width assert that had no equivalent upstream

`synthetic_seg_test.rs`'s local `BitWriter::push_bits` had one line with no counterpart in
`src/header.rs`'s original: `assert!(n == 32 || value >> n == 0, "{value} does not fit in
{n} bits")`, added there specifically to fail loudly if a hand-encoded field value like an
`ALT_L` level were to overflow its bit width and silently truncate into a different valid
encoding. The wave's instructions describe the three copies as "verbatim" and direct
replacing this one with the shared `vp9dec::test_support::BitWriter`, which (matching the
`header.rs` original that was actually moved) has no such assert. Replaced as instructed
rather than carrying the extra assert into the shared version, since none of the current
callers in either file encode a value that overflows its field (`cargo test` stayed
green), but flagging here because the failure mode this assert guarded against (silent
truncation into a different, still-valid encoding) is exactly the kind of bug that fails
far from its cause -- a future editor adding a hand-encoded field to either test file no
longer gets that check for free.

### IVF writer unification

Added `pub fn write_ivf(fourcc, width, height, timebase_den, timebase_num, frames) ->
Vec<u8>` to `src/ivf.rs` (the inverse of `IvfReader`, timestamps = frame index), and
migrated the three existing hand-rolled writers to it:

- `examples/webm_to_ivf.rs`'s `write_ivf` now calls the shared one with `1, 1` for the
  timebase, preserving that example's existing output byte-for-byte (it was already
  hard-coding `1/1` regardless of the source WebM's actual timebase, a pre-existing
  behavior this wave didn't change).
- `tests/synthetic_seg_test.rs`'s `ivf_wrap` helper is gone; its two call sites now call
  `vp9dec::ivf::write_ivf(b"VP90", WIDTH as u16, HEIGHT as u16, 30, 1, &frames)` directly,
  same `30/1` timebase as before.
- `src/ivf.rs`'s own `#[cfg(test)] build_file_header`/`append_frame` helpers were only
  *partially* retired: `parses_file_header_fields`, `rejects_bad_signature`, and
  `handles_empty_stream` now build their input via the new `write_ivf`, but
  `iterates_frames_in_order` (needs non-sequential timestamps `0, 33, 66` to prove
  timestamps are read correctly, which `write_ivf`'s "timestamp = index" contract can't
  produce) and `reports_truncated_frame_data` (deliberately appends a frame header
  claiming more data than actually follows) still need hand-built bytes, so
  `build_file_header`/`append_frame` stay for those two.

### `Cargo.toml` / example test wiring

Added `publish = false`, the `test-support` feature, the self dev-dependency, and explicit
`[[example]]` sections for both `decode_to_png` (`test = true`) and `webm_to_ivf` (no
`test = true` needed -- it has no `#[cfg(test)]` tests of its own). Declaring one
`[[example]]` section did not disable auto-discovery of the other; both continued to be
picked up and build correctly (`cargo build --examples`), which was verified rather than
assumed. `test = true` brought `decode_to_png.rs`'s four pre-existing `#[cfg(test)]` tests
(crc32/adler32/deflate/PNG-chunk checks) into `cargo test` for the first time; confirmed
via `cargo test` output that all four actually ran and passed, raising the total from 146
to 150.

### Judgment call: `rustfmt <file>` on the crate root reformats the whole crate

The wave's instructions say to format only touched files via `rustfmt <file>` rather than
bare `cargo fmt` (which reformats pre-existing drift in unrelated files -- a known
incident). Running `rustfmt --check` file-by-file surfaced a sharp edge in that plan:
`src/lib.rs` is the crate root, and rustfmt follows `mod` declarations from whichever file
it's given, so `rustfmt src/lib.rs` reports (and, run for real, would rewrite) diffs
across every module reachable from it -- functionally identical to bare `cargo fmt` for
this crate, just invoked differently. Resolution: `rustfmt --check` was run per touched
file to *detect* issues, but only applied by hand (via targeted edits, not `rustfmt` as a
formatter) to the specific hunks that trace to this wave's own edits -- an import-order
fix each in `src/compressed_header.rs` and `src/tile.rs` (inserting `use
crate::test_support::BoolEncoder;` out of alphabetical order), and a line-wrap each in
`examples/webm_to_ivf.rs` and `tests/synthetic_seg_test.rs` (lines that grew past the
column limit). Pre-existing drift the `--check` run also reported in these and other
touched files (`src/header.rs`, `src/lib.rs`, `src/prob_tables.rs`, `src/tile.rs`,
`tests/compressed_header_test.rs`, and the bulk of `examples/webm_to_ivf.rs`) was left
untouched, matching "surgical changes" over the letter of "format files you touched."
`src/lib.rs`'s own one-line diff addition (`pub mod test_support;`) was verified by
inspection to already match rustfmt's expected style, so no reformatting was needed there
in any form.

### Verification

`cargo test`: 150 passed (125 lib unit + 2 + 7 + 2 + 2 + 2 + 6 integration + 4 example),
0 failed, across all 9 binaries. `cargo test --test synthetic_seg_test -- --nocapture`:
6/6, only the two expected `[skip]` lines (`VP9DEC_DUMP_DIR` unset, ffmpeg not on PATH).
`cargo build` (no features): succeeds without compiling `test_support`. `cargo clippy
--all-targets`: only the same 3 pre-existing warnings (`src/header.rs`
`large_enum_variant`, `src/superframe.rs` `identity_op`, `src/lib.rs`
`field_reassign_with_default`); the two `new_without_default` warnings clippy raised
against the newly-`pub` `BoolEncoder`/`BitWriter` were fixed by adding `Default` impls
(delegating to `new()`) rather than suppressed.

## Wave 2a: internal signature redesign (2026-07-16)

### Scope

Killed the "cross-frame state threaded one-parameter-per-milestone" pattern and the `_ex`
suffix accretion, purely internal (public `Decoder`/`decode_keyframe` behavior is
byte-identical; all 7 MD5 conformance tests + 4 synthetic round-trip tests pin it
unmodified). Touched `src/header.rs`, `src/compressed_header.rs`, `src/lib.rs`,
`src/tile.rs`, `tests/header_test.rs`, `tests/compressed_header_test.rs`.
`tests/synthetic_seg_test.rs` calls only `Decoder`'s public API and needed no changes
(verified by inspection: no direct `parse_uncompressed_header`/`parse_compressed_header`/
`TileDecoder` calls in that file).

### `PersistentState` (spec §7.2 cross-frame state)

`src/header.rs` gained three new structs: `LoopFilterDeltas { ref_deltas, mode_deltas }`,
`SegFeaturePersist { enabled, data, abs_or_delta }` (replacing the former
`(SegFeatureEnabled, SegFeatureData, bool)` tuple), and `PersistentState {
ref_frame_sizes, loop_filter_deltas, segmentation }` bundling both plus the reference
frame size table. All three derive `Copy` (matching the tuples they replace) and get a
`new()`/`Default` pair whose value is exactly `Decoder::new()`'s old field-by-field
initialization (`ref_frame_sizes` all-zero, `loop_filter_deltas` = the spec's
`DEFAULT_LOOP_FILTER_*` constants -- NOT all-zero, despite the task brief's "all-zero
state" phrasing being a simplification -- `segmentation` all-zero/false). This matters
because `decode_keyframe`'s old "dummy" values were built from exactly those same
constants; replacing them with `PersistentState::default()` is only behavior-preserving
because the default reproduces that exact prior value, not a literal all-zero struct.

`parse_uncompressed_header(data, prev: &PersistentState)` replaces the old 4-parameter
form; `parse_loop_filter_params`/`parse_segmentation_params` take the new named structs
(by value, since `Copy`) instead of tuples. `Decoder` now holds one `persist:
PersistentState` field instead of `ref_frame_sizes`/`loop_filter_deltas`/
`segmentation_features`; all three post-decode write-back sites in `decode_one_frame`
write into `self.persist.*` instead.

### Honest `color_config`

`NewFrameHeader::color_config` is now `Option<ColorConfig>`: `Some` for key frames
(always) and `intra_only` frames (both the `profile > 0` parsed case and the `profile ==
0` spec-defined 8-bit/CS_BT_601/4:2:0 default -- confirmed the latter is genuinely
spec-defined, not a fabrication, by re-reading spec §6.2.2's `color_config()` default
path before leaving that branch `Some`); `None` for a regular inter frame, which the old
code filled with a fabricated `{CS_UNKNOWN, 8-bit, 4:2:0}` placeholder.

`Decoder::decode_one_frame` resolves this once, right after the `FrameHeader::New` match:

```rust
let color_config = header.color_config.unwrap_or_else(|| {
    self.last_color_config.unwrap_or(ColorConfig { bit_depth: 8, color_space: CS_UNKNOWN,
        color_range: false, subsampling_x: 1, subsampling_y: 1 })
});
if header.frame_is_intra {
    self.last_color_config = Some(color_config);
}
```

Proved this behavior-neutral by case analysis rather than by trusting the tests alone
(the degenerate branch -- an inter frame before any intra frame ever ran -- isn't
exercised by any conformance vector, so a test pass wouldn't have caught a subtle
mismatch here):

- `frame_is_intra == true` (key or intra_only): `header.color_config` is always `Some`,
  so `unwrap_or_else`'s closure never runs; `color_config` is exactly the parsed/
  spec-default value, and `last_color_config` is refreshed to it -- identical to the old
  `self.last_color_config = Some(header.color_config)`.
- `frame_is_intra == false` with a prior intra frame seen: `header.color_config` is
  `None`, so `color_config` = `self.last_color_config.unwrap()` -- identical to the old
  `else if let Some(cc) = self.last_color_config { header.color_config = cc; }`.
- `frame_is_intra == false`, no prior intra frame (`last_color_config` still `None`,
  malformed-but-defensively-handled stream): `color_config` falls through to the
  hardcoded `{8-bit, CS_UNKNOWN, 4:2:0}` literal -- byte-for-byte the same struct
  `header.rs` used to fabricate inline for every non-intra frame, just now materialized
  in `lib.rs` only for this one fallback case instead of unconditionally.

The resolved `color_config` then replaces every downstream read of `header.color_config`
(the bit-depth check, `build_ref_frame_data`, `crop_to_frame`, and the new
`TileDecoder::new`/`new_with_prev` parameter below) -- `header.color_config` itself is
never read again after the resolution point.

### `TileDecoder::new`/`new_with_prev` gained a `color_config: ColorConfig` parameter

Not explicitly requested by the task brief, but a necessary consequence of making
`color_config` honest: `TileDecoder::new_with_prev` previously read
`header.color_config.bit_depth`/`.subsampling_x`/`.subsampling_y` directly, which no
longer type-checks once the field is `Option`. Rather than have `TileDecoder` re-derive
its own fallback (reintroducing exactly the fabrication this wave removes, just
relocated), it now takes the caller's already-resolved `ColorConfig` as a plain
parameter, inserted right after `header`. This rippled into ~14 call sites: `src/lib.rs`
(1), `src/tile.rs`'s own unit tests (12, all via `header.color_config.unwrap()` since
their `minimal_header()` test helper always sets it to `Some`), and
`tests/compressed_header_test.rs` (1, via `header.color_config.expect(...)` since the
first frame of any IVF stream is a key frame).

### Collapsed `parse_compressed_header`/`_ex`

`parse_compressed_header(data, header: &NewFrameHeader, starting_probs: FrameContext)`
replaces both the old keyframe-only wrapper and the 7-argument `_ex` (`#[allow(
clippy::too_many_arguments)]` deleted along with it -- `header.rs` had no such allow to
begin with). Verified all five of `_ex`'s former loose parameters
(`lossless`/`frame_is_intra`/`interpolation_filter`/`ref_frame_sign_bias`/
`allow_high_precision_mv`) exist on `NewFrameHeader` already, so no field needed adding.
`compressed_header.rs`'s 5 internal unit tests and `tests/compressed_header_test.rs`'s
external one previously called the keyframe-only wrapper (which hardcoded
`frame_is_intra = true`, `interpolation_filter = SWITCHABLE`, `ref_frame_sign_bias =
[false; 4]`, `allow_high_precision_mv = false`); a `#[cfg(test)] fn key_frame_header
(lossless) -> NewFrameHeader` helper in `compressed_header.rs` now builds a full header
carrying those same four values (they match what a real key frame's
`parse_uncompressed_header` produces anyway, since those fields are never touched by the
key-frame branch), so the encoded bitstreams in those tests needed no changes.

### Judgment call: recovering from an accidental `rustfmt src/lib.rs` invocation

Ran `rustfmt --check` in a shell loop over all six touched files for the final
verification pass, including `src/lib.rs` -- forgetting, in the moment, that the
project's standing rule against `rustfmt src/lib.rs` applies even in `--check` mode
(module-tree cascading is triggered by rustfmt's file-discovery logic regardless of
`--check` vs. write). No files were written (`--check` only reports), but the output
confirmed the cascade did happen: diffs were reported against `src/compressed_header.rs`,
`src/header.rs`, `src/prob_tables.rs`, and `src/lib.rs` itself from that single `src/
lib.rs` invocation. Recovery: abandoned that loop entirely; ran `rustfmt` (write mode)
individually only on the leaf-module files with no `mod x;` file-declarations of their
own (`src/compressed_header.rs`, `src/header.rs`, `src/tile.rs`,
`tests/compressed_header_test.rs`, `tests/header_test.rs` -- each only has inline
`#[cfg(test)] mod tests { ... }`, so formatting them individually cannot cascade
further), then hand-fixed `src/lib.rs`'s two over-length lines introduced by this wave's
own edits via targeted `Edit` calls, leaving the rest of `src/lib.rs` (and
`src/prob_tables.rs`, untouched by this wave at all) alone.

One of those individual `rustfmt` invocations swept up unrelated pre-existing drift
anyway: formatting `src/tile.rs` as a whole file also rewrapped `read_ref_frames`'s
signature (a function this wave never touched) onto multiple lines. Caught this by
diffing before/after and reverted that one hunk by hand, restoring the original one-line
signature -- `rustfmt --check src/tile.rs` now reports exactly that one pre-existing
diff, left deliberately alone per "leave unrelated hunks alone."

### Verification

`cargo test`: 150 passed, 0 failed, across all 9 binaries -- unchanged from the Wave 1
baseline (125 lib unit + 2 + 7 + 2 + 2 + 2 + 6 integration + 4 example). Conformance
(`cargo test --test conformance_test -- --nocapture`): 7/7, all printing real `[ok]`
lines with vector paths (none skipped). `VP9DEC_FFMPEG=... cargo test --test
synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`: all 8
`[xdecode] .../libvpx-vp9: OK` / `.../vp9: OK` lines present. `cargo clippy
--all-targets`: the same 3 pre-existing warnings as Wave 1's baseline
(`src/header.rs` `large_enum_variant`, `src/superframe.rs` `identity_op`, `src/lib.rs`
`field_reassign_with_default`, all confirmed via `git diff` to fall on lines this wave
never touched); the `too_many_arguments` allow removed from `compressed_header.rs` did
not reappear (the new 3-parameter `parse_compressed_header` is well under the default
threshold), and `tile.rs`/`loop_filter.rs`/`predict.rs`'s pre-existing allows for that
lint were left untouched as instructed. `git diff --stat`: exactly the 6 intended files
(`src/header.rs`, `src/compressed_header.rs`, `src/lib.rs`, `src/tile.rs`,
`tests/header_test.rs`, `tests/compressed_header_test.rs`); `tests/synthetic_seg_test.rs`,
`tests/decode_test.rs`, `tests/inter_frame_test.rs`, and `tests/conformance_test.rs` show
no diff.

## Wave 2b: public API redesign -- per-constituent decode results (2026-07-16)

Breaking change replacing the "one chunk -> at most one visible `Frame`" contract with
per-constituent-frame results, plus deletion of the M2-era legacy API and a narrowing of
the public surface. Decoded pixels are unchanged (pinned by the all-frames MD5
conformance tests and the ffmpeg cross-decode, all green before and after).

### New shape

- `Decoder::decode_frame(chunk) -> Result<Vec<DecodedFrame>, DecodeError>`: one element
  per constituent VP9 frame of the chunk (superframes yield several), in bitstream
  order. `DecodedFrame { info: Option<FrameDecodeInfo>, frame: Option<Frame> }` --
  `info` is `None` only on the `show_existing_frame` path (no uncompressed header is
  parsed there, so no stats exist); `frame` is `None` for hidden (`show_frame == 0`)
  frames. A conformant chunk has at most one `frame: Some` element, deliberately not
  enforced: the decoder reports what happened rather than policing stream conformance
  at the API boundary.
- Deleted: `decode_keyframe()`, `DecodeError::NotAKeyFrame`, `Decoder::last_frame_info()`
  and its backing field. Info traveling in the return value also removes the old wart
  where a `show_existing_frame` chunk left the *previous* frame's info observable.
- Every `pub mod` except `ivf` is now `#[doc(hidden)]` (internal; public only so the
  pure-std integration tests can reach them). The documented public surface is
  `Decoder`, `DecodedFrame`, `Frame`, `FrameDecodeInfo`, `DecodeError`, and `ivf`.
  All five types are defined in `lib.rs` itself, so no `pub use` re-exports were needed.

### Test suite changes (150 -> 148 tests)

- Deleted `tests/conformance_test.rs`'s `check_vector` helper and its two tests
  (`vp90_2_12_droppable_1_first_keyframe_matches_official_md5`,
  `vp90_2_09_subpixel_00_first_keyframe_matches_official_md5`): strictly subsumed by the
  `*_all_frames_match_official_md5` tests over the same vectors (frame 1 is the first
  line those tests compare anyway). Pre-authorized by the wave plan.
- `check_all_frames_with_coverage`'s manual `split_superframe` loop (which existed only
  because `last_frame_info()` was per-chunk) collapsed onto the per-constituent API;
  coverage strength is preserved because every returned `DecodedFrame` contributes its
  `info` (hidden sub-frames included -- vp90-2-16-intra-only still reports
  `reset_frame_context values seen = {0, 2}` and its `intra_only` predicate passes on
  the hidden constituents).
- `src/lib.rs`'s `last_frame_info_reflects_a_decoded_keyframe` rewritten as
  `decoded_frame_info_reflects_a_decoded_keyframe`: same three header assertions read
  from the returned `DecodedFrame`, plus new `len() == 1` / `frame.is_some()` checks.
  The old "None before the first decode" assertion has no equivalent (the concept of
  pre-decode info state no longer exists) -- that is the API change itself, not a lost
  check.
- `tests/decode_test.rs` kept (not deleted): the Y-plane statistics plausibility test
  adapts trivially to `Decoder` (`find_map(|df| df.frame)` on the first chunk), and it
  is the only test asserting those statistics on real vectors.
- `tests/inter_frame_test.rs`: minimal mechanical adaptation only (scheduled for
  deletion in Wave 3). `hidden_count` in its log line now counts hidden *constituent*
  frames rather than all-hidden *chunks*; its assertions never used it.

### Judgment calls

- `README.md` still documents `decode_keyframe()` in its M2/M2b milestone narrative
  (~9 mentions). Left untouched: the wave's task list scoped consumers to code +
  rustdoc, and the README sections are historical milestone descriptions. Follow-up
  candidate for the next docs pass.
- Stale rustdoc references fixed alongside the deletions: `src/header.rs`
  (`PersistentState` doc), `src/tile.rs` x2 (now point at [`crate::Decoder`]),
  `src/loop_filter.rs` (now cites `DecodeError::UnsupportedBitDepth` for the 8-bit
  limit), `tests/compressed_header_test.rs` module doc. `cargo doc --no-deps` with
  `-D rustdoc::broken_intra_doc_links` passes; the pre-existing
  `private_intra_doc_links` warnings dropped 3 -> 2 (the old `decode_frame` doc linked
  to the private `decode_one_frame`; the remaining 2 are `src/tile.rs` module-doc links
  untouched by this wave).
- `rustfmt` applied only to `tests/synthetic_seg_test.rs` (the one touched leaf file
  with rustfmt-relevant diffs, both in newly written lines). `src/tile.rs` and
  `examples/decode_to_png.rs` were left unformatted because their only reported diffs
  are pre-existing drift outside this wave's hunks (`read_ref_frames`' one-line
  signature, kept verbatim per the Wave 2a note, and the example's `target_index`
  arg-parsing closure). `src/lib.rs` edits hand-formatted; no added line exceeds the
  100-column limit.

## Wave 3: test-layer consolidation (2026-07-16)

### Scope

Collapsed the milestone-accreted test layer -- 6 integration test files, each with its own
copy of vector-loading/skip/md5 boilerplate -- down to 3, with the shared infrastructure
factored into `tests/common/`.

### What moved where

- **`tests/common/mod.rs`** (new): `vectors_dir()`/`read_vector()`/`read_vector_with_md5()`/
  `first_ivf_frame()`/`i420_bytes()`, extracted from what were 8 duplicated skip-if-absent
  blocks, 8 vectors-path constructions, 3 I420-concat copies, and 3 `.ivf.md5` parsers spread
  across the 6 original files. The module head carries `#![allow(dead_code)]` (cascades to its
  `encoder`/`md5` submodules) since each `tests/*.rs` binary recompiles this module and uses
  only a subset of it -- e.g. `api_test.rs` never touches `i420_bytes` or `md5`.
- **`tests/common/md5.rs`**: relocated verbatim from `src/md5.rs` (the RFC 1321 implementation);
  `pub mod md5;` removed from `src/lib.rs` since its sole external consumer was
  `conformance_test.rs`. Its 7 unit tests moved into `tests/conformance_test.rs`'s
  `mod md5_tests` (option (a) from the task) rather than staying alongside the implementation in
  `tests/common/md5.rs`: everything under `tests/common/` recompiles once per consuming test
  binary, so a `#[test]` there would rerun once per binary instead of once overall.
- **`tests/common/encoder.rs`** (new): the ~400-line synthetic VP9 bitstream encoder
  (`SegSpec`, the header/tile/compressed-header builders, `tree_path`/`encode_tree`,
  `kb`/`header_size`/`assemble_frame`, `FEATURE_BITS`/`FEATURE_SIGNED`, `WIDTH`/`HEIGHT`)
  extracted from `tests/synthetic_seg_test.rs`, so a future synthetic-vector test file (e.g. M4's
  planned reference-frame-scaling coverage) can reuse it without re-deriving ~400 lines of
  encoder plumbing. The scenario builders (`build_skip_ref_frames`/`build_steering_frames`/
  `build_alt_l_frames`/`scenarios()`) and all 6 `#[test]` fns stayed in
  `synthetic_seg_test.rs` per the task's default -- judged not worth moving further, since
  they're this file's actual test content (what varies per scenario), not shared plumbing.
- **`tests/conformance_test.rs`**: `check_all_frames` and `check_all_frames_with_coverage`
  (~80% identical) merged into one `check_all_frames(ivf_name, coverage: Option<Coverage>)`,
  `type Coverage = (&'static str, fn(&FrameDecodeInfo) -> bool)` (a type alias only to satisfy
  clippy's `type_complexity` lint on the bare tuple). `FrameDecodeInfo` is now collected
  unconditionally in the decode loop (previously only in the `_with_coverage` copy) -- this is
  bookkeeping, not an extra assertion, so it changes nothing the 2 non-coverage callers
  (`droppable_1`/`subpixel_00`) check. The 3 coverage callers (`segkey`/`segkey_adpq`/
  `intra-only`) keep their exact predicate and `eprintln!` coverage summary.
- **`tests/inter_frame_test.rs`**: deleted (pre-authorized). Its "every frame parses without
  panic" check over the same two vectors is strictly subsumed by `conformance_test.rs`'s
  all-frames MD5 check (which also decodes every frame in order, and additionally verifies
  pixel-exact output), and it double-decoded 119 frames on every `cargo test` run for no
  incremental coverage.
- **`tests/api_test.rs`** (new): `header_test.rs` + `compressed_header_test.rs` +
  `decode_test.rs` merged verbatim (same test names, same assertions; their 3 `check_vector`
  helpers renamed to `check_header_vector`/`check_compressed_header_vector`/
  `check_decode_vector` to avoid collision in one file). The three originals deleted.

### Judgment calls

- `common::first_ivf_frame(bytes: &[u8]) -> (IvfHeader, &[u8])` replaced the repeated
  "`IvfReader::new` + `.next()`" boilerplate common to all 3 `api_test.rs` probes (and used by
  `check_header_vector` for the fourcc/width/height check). Returns the owned `IvfHeader`
  (`Clone`) alongside a `&[u8]` borrowing from the input, rather than the `IvfReader` itself,
  since none of the 3 callers need to continue iterating past the first frame.
- `SegSpec`'s fields and `KeyBlock` became `pub` (mostly `pub` fields) purely as a mechanical
  consequence of the file split -- `synthetic_seg_test.rs`'s scenario builders mutate `SegSpec`
  fields directly (e.g. `seg.feature_enabled[0][SEG_LVL_SKIP] = true`) and now live in a
  different module than the struct definition. No behavior change.
- Left `#[cfg(test)]` off the relocated md5 unit tests (`mod md5_tests` in
  `conformance_test.rs`): every `tests/*.rs` file is only ever compiled under `cargo test`'s
  implicit `--test` flag (which itself implies `--cfg test`), so the gate would be
  permanently-true dead weight there, unlike its original home in `src/md5.rs` where it
  distinguished test builds from the shipped library.

### Verification

`cargo test`: 146/146 pass across 5 binaries -- lib 118, `api_test` 6, `conformance_test` 12
(5 conformance + 7 relocated md5 unit tests), `synthetic_seg_test` 6, `decode_to_png` example 4.
This is exactly 148 (the pre-Wave-3 total across the former 8 binaries/files) minus
`inter_frame_test.rs`'s 2 deleted tests, with no other count changes anywhere. Conformance: all
5 tests print real `[ok]`/`[coverage]` lines against real vector paths (not skipped); the skip
path was re-verified by temporarily renaming `tests/vectors/` (produced the expected `[skip]`
line, the test still passed, the directory was renamed back immediately after).
`VP9DEC_FFMPEG=... cargo test --test synthetic_seg_test
synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`: 8/8 `[xdecode]` OK lines
(unchanged from before this wave -- `encoder.rs`'s extraction is a pure move, no byte-level
behavior changed). `cargo clippy --all-targets`: 3 pre-existing warnings only
(`large_enum_variant` in `src/header.rs`, `identity_op` in `src/superframe.rs`,
`field_reassign_with_default` in `src/lib.rs`); one new `type_complexity` warning surfaced by
the merged `check_all_frames`'s bare tuple-`Option` parameter, fixed with the `Coverage` type
alias above before being recorded as clean.
`RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps`: clean pass (2
pre-existing `private_intra_doc_links` warnings in `src/tile.rs`, unrelated to the md5 move --
nothing publicly documented ever linked to `vp9dec::md5`). `rustfmt --check` per touched leaf
file: only `tests/api_test.rs` needed reformatting (a line over 100 columns produced by the
merge), applied; `tests/common/{mod,md5,encoder}.rs`, `tests/conformance_test.rs`, and
`tests/synthetic_seg_test.rs` were already clean. `src/lib.rs`'s 2-line mod-declaration removal
was not run through rustfmt (per the standing constraint) and needed no reformatting regardless.

## Wave 4a: performance -- eliminate per-frame deep-clone traffic (2026-07-16)

### Scope

Removed the per-frame deep-clone traffic identified by the architecture review: DPB slot
writes (up to 8x per keyframe), per-frame reference resolution, the `MiGrid`/`PrevSegmentIds`
clone-in/clone-out, and the `CompressedHeaderProbs` clone into every `TileDecoder`. Bit-exact
output was the hard constraint throughout -- every change is a sharing-strategy change
(`Arc`), never a logic change.

### 1. `Dpb` slots -> `Arc<RefFrameData>` (`src/dpb.rs`, `src/lib.rs`, `src/tile.rs`)

`Dpb::slots` is now `[Option<Arc<RefFrameData>>; NUM_REF_FRAMES]`. `Dpb::get` (used only by
the `show_existing_frame` path) keeps its `Option<&RefFrameData>` signature via `as_deref()`,
so that one call site is untouched. A new `Dpb::get_arc(idx) -> Option<Arc<RefFrameData>>`
(a refcount bump, `.clone()` on the `Option<Arc<_>>` slot) serves `Decoder::decode_one_frame`'s
`resolved_refs` resolution, replacing `self.dpb.get(idx).cloned()`. `Dpb::update` now takes
`data: &Arc<RefFrameData>` and stores `data.clone()` (Arc clone) per set bit in
`refresh_frame_flags`, instead of a full `RefFrameData::clone()` per bit; the caller builds the
frame's pixel data once (`Arc::new(build_ref_frame_data(...))`) regardless of how many slots get
refreshed. `TileDecoder::resolved_refs` became `[Option<Arc<RefFrameData>>; 3]`; its one read
site (building `RefPlaneView`s for `predict_inter`) switched `.as_ref()` to `.as_deref()` to keep
producing `Option<&RefFrameData>`. `RefFrameData`'s `Clone` derive was dropped (nothing clones
the struct itself anymore -- only the `Arc` around it), keeping `Debug`.

### 2. `prev_mi_grid` / `prev_segment_ids`: stop the clone-in/clone-out (`src/lib.rs`, `src/tile.rs`)

Chose approach (a) from the task (Arc-shared, not restore-on-error): `Decoder::prev_mi_grid`
is `Option<Arc<MiGrid>>`; the clone-in (`self.prev_mi_grid.clone()` when `use_prev_frame_mvs`)
is now an `Arc` clone. `TileDecoder::prev_mi_grid` became `Option<Arc<MiGrid>>` -- its only use
(`get_block_mv`, spec's `usePrev` branch) is a read through `.get(row, col)`, unaffected by the
extra indirection; verified by inspection (and by the two-argument search across the file) that
nothing else touches `prev_mi_grid`, so no mutation-through-`Arc` hazard exists. `MiGrid`'s
`Clone` derive was dropped (it was only ever cloned at the two sites removed here) in favor of
just `Debug`.

The clone-out (`self.prev_mi_grid = Some(tile_decoder.mi_grid().clone())`) is eliminated via a
new consuming `TileDecoder::into_mi_grid(self) -> MiGrid` (moves `self.mi_grid` out, no copy).
This must be the *last* thing done with `tile_decoder`, so the call was relocated from
immediately after `decode_tiles`/`apply_loop_filter` to the very end of `decode_one_frame`,
after the final `tile_decoder.planes()` call inside the `show_frame` branch (the only other
remaining use of `tile_decoder` past that point). `mi_grid()` (the borrowing accessor) is
unchanged and still serves the earlier `PrevSegmentIds`-refresh read.

Per the task's explicit landmine warning, `Option::take()` was not used anywhere: the read side
is a `clone()` (cheap now, but still non-destructional), so `self.prev_mi_grid` is left exactly
as-is if `decode_tiles` errors out via `?` before the function reaches the write side -- there
is no window where it's transiently `None`/stale relative to `prev_frame_dims`/`prev_show_frame`.
See the new regression test below.

`Decoder::prev_segment_ids` / `TileDecoder::prev_segment_ids` got the same treatment
(`Arc<Vec<u8>>`): unlike `prev_mi_grid` this clone-in is unconditional every frame (not gated by
a `use_prev_*` flag), and `TileDecoder` has exactly one read site (`get_segment_id`'s
`prev_segment_ids[idx]`, indexing through the `Arc` via auto-deref), so it was cheap to include
and cuts a ~`MiRows*MiCols`-byte clone every single frame regardless of segmentation being
in use. The write side already built a fresh `Vec` (never cloned an existing one), so this is
just a wrapping change (`Arc::new(new_map)` / `Arc::new(vec![0u8; ...])`), no restructuring.

### 3. `CompressedHeaderProbs` clone reduction (`src/compressed_header.rs`, `src/lib.rs`, `src/tile.rs`)

`CompressedHeader::probs` is now `Arc<CompressedHeaderProbs>` (wrapped once, at the end of
`parse_compressed_header`). `TileDecoder::probs` followed suit (`Arc<CompressedHeaderProbs>`);
its constructor line (`probs: compressed.probs.clone()`) is textually unchanged but now an Arc
clone instead of a multi-KB deep clone (confirmed by inspection: zero write sites to
`self.probs` anywhere in `tile.rs`, only field reads through the automatic `Deref`).

Kept exactly as pre-validated:

- `FrameContextStore::load` still returns an owned `CompressedHeaderProbs` (unrelated to the
  `Arc`; the store's slots are still bare `CompressedHeaderProbs`, never `Arc`, so a later
  frame's forward-update on its `load`ed copy can never alias/mutate a stored slot).
- `starting_probs.clone()` fed into `parse_compressed_header` is untouched (it's forward-updated
  by value inside that call; the original is still needed afterward by `refresh_probs`).
- The error-resilient/frame-parallel branch of `refresh_probs` still does a full deep clone --
  now spelled `(*compressed.probs).clone()` rather than `compressed.probs.clone()`, because the
  latter would resolve to `Arc::clone` (an `Arc<CompressedHeaderProbs>`, wrong type for
  `FrameContextStore::save`, which stores owned contexts) rather than a deep copy of the probs.

Eliminated: the `let mut working = starting_probs.clone();` at the top of the normal
`refresh_probs` branch. Since `starting_probs.tx_probs`/`starting_probs.skip_prob` are read
again later in that same branch (`load_probs2`'s restore, only under `!frame_is_intra`), those
two fields (24B + 3B, both `Copy`) are copied aside into locals *before* `starting_probs` is
moved (not cloned) into `working`; the later restore reads the locals instead of `starting_probs`.
Confirmed by full-function search that `starting_probs` has no other use after this point, so
the move is sound. This is the one genuinely "clever" mechanic in this wave -- everything else
is a sharing-strategy swap.

### Judgment calls

- `prev_segment_ids` was explicitly called optional by the task ("if cheap"); did it anyway
  since `TileDecoder` has a single read site and the write side needed no restructuring --
  see above.
- Named the consuming accessor `into_mi_grid` rather than a more general `into_parts()` (which
  the task described as a style, not a mandate): `MiGrid` is the only piece of `TileDecoder`
  state that needs move-out-without-clone treatment (`planes()`/`counts()` etc. are all still
  read by reference before the final call), so a single-purpose method avoids speculative
  generality for a caller that doesn't exist yet.

### Verification

Added `tests::decode_recovers_after_a_mid_frame_tile_error` (`src/lib.rs`): decodes a real
key frame from `vp90-2-12-droppable_1.ivf`, then feeds the second frame truncated to 29 bytes
(empirically the exact byte count that leaves the uncompressed+compressed headers parseable
but starves `decode_tiles`, which fails with `Tile(BoolCoder(EmptyBuffer))` -- verified by
bisection before hardcoding), asserts that specific error, then decodes the (untouched) third
frame and asserts success. This is the regression coverage for the task 2 landmine: with an
`Arc`-clone (not `take()`) implementation there is no window where `prev_mi_grid` and
`prev_frame_dims`/`prev_show_frame` disagree, so the third frame must decode without panicking.

`cargo test`: 147/147 pass (146 from Wave 3 + 1 new regression test above; no other count
change) across all 6 binaries/targets. Conformance: all 5 vector tests print real `[ok]` lines
against real vector paths (unchanged bit-exact output -- this wave changes ownership, not
values). `VP9DEC_FFMPEG=... cargo test --test synthetic_seg_test
synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`: 8/8 `[xdecode]` OK lines.
`cargo clippy --all-targets`: the same 3 pre-existing warnings as Wave 3 (confirmed identical
via `git stash`), no new ones. `rustfmt --check` per touched leaf file (`dpb.rs`,
`tile.rs`, `compressed_header.rs`; `lib.rs` deliberately never run through rustfmt, hand-
formatted instead): only one pre-existing unrelated diff remains in `tile.rs`
(`read_ref_frames`'s over-100-column signature, confirmed present at HEAD before this wave via
`git stash` -- left untouched per the surgical-changes rule). `git diff --stat`: exactly the
4 expected files (`dpb.rs`, `lib.rs`, `tile.rs`, `compressed_header.rs`; `predict.rs` needed no
changes since its `RefPlaneView<'a>` already borrows a `&'a Plane` obtained after dereferencing,
unaffected by the `Arc` wrapping upstream).

Timing (`cargo test --release --test conformance_test 2>&1 | tail -3`, this machine, clean
`cargo clean --release -p vp9dec` before each run to force a full rebuild+run; the number
quoted is the suite's own "finished in Xs" line, i.e. run time, not compile time): before
2.32s, after 2.22s. Debug profile (`cargo test --test conformance_test`): before 48.42s, after
48.32s. Both are within noise for this suite -- the 5 conformance vectors are small test clips
(largest is 352x288), so the per-frame `MiGrid`/`RefFrameData` clones this wave removes are on
the order of hundreds of KB - low single-digit MB each, not enough to dominate wall time
against bool-decoding/transform/loop-filter work at these resolutions. The wave's value is
architectural (removing clone traffic that scales with frame area, ahead of the eventual Noiria
integration with real-world 1080p+ sources where the same clones would cost proportionally
more), not a measured win on today's conformance suite.

## Wave 4b: performance -- hot-path allocations + per-frame dequant derivation (2026-07-16)

### Scope

Removed per-block heap allocations from the decode/reconstruction hot paths and hoisted the
per-block dequant-index derivation (`get_qindex`/`get_dc_quant`/`get_ac_quant`) into a table
built once per frame. Every change is a storage-strategy change (`Vec` -> fixed-size array,
sized to the spec's hard maxima) or a hoist-a-pure-function-of-frame-state-out-of-the-loop
change -- no arithmetic expression was reordered or altered; bit-exactness was the hard gate
throughout (verified per-step via the full test suite, not just at the end).

### 1. `src/transform.rs`: per-row/column `Vec`/`.to_vec()` -> fixed `[i64; N]`

`idct_permute` (`.to_vec()` of up to `n0` elements, `n0` <= 32 since `n` <= 5) and
`adst_input_permute`/`adst_output_permute` (`n0` <= 16, ADST tops out at 16x16) each did a
`.to_vec()` heap allocation *per call*, and both are called once per row and once per column of
every transform block -- the single biggest allocation-count contributor in the decoder.
Replaced with a local `[i64; 32]` / `[i64; 16]` stack scratch, copied into via
`copy_from_slice` instead of `to_vec()`. `inverse_transform_block`'s own per-block
`vec![0i64; n0]` row/column scratch became a `[i64; 32]` local (`t_storage`) sliced to
`&mut t_storage[..n0]`; the call sites that used to pass `&mut t` (a `Vec`, auto-deref to
`&mut [i64]`) now pass `t` directly (already `&mut [i64]`) -- Rust's implicit-reborrow rule for
`&mut` references used directly as a call argument makes this a non-issue across the loop's
repeated calls, no explicit reborrowing syntax needed.

### 2. `src/tile.rs::tokens_and_reconstruct`: `tokens`/`dequant` -> fixed `[_; 1024]`

`tokens: vec![0i32; seg_eob]` and `dequant: vec![0i64; n0*n0]` (two heap allocs per transform
block) became `[0i32; 1024]` / `[0i64; 1024]` locals (1024 = 32*32, the 32x32 max transform),
sliced to `[..seg_eob]` where the code needs the exact logical length (the iteration over
`tokens` when building `dequant`, and the slice handed to `inverse_transform_block`, which
asserts `dequant.len() == n0*n0`). The final reconstruction loop indexes `dequant[i*n0+j]`
directly against the full fixed array, which is safe unchanged since `i*n0+j < n0*n0 <= 1024`.

### 3. Per-frame dequant table (`src/tile.rs`)

Added a free function `build_dequant_table(segmentation, base_q_idx, bit_depth, delta_q_y_dc,
delta_q_uv_dc, delta_q_uv_ac) -> [[[i64; 2]; 2]; MAX_SEGMENTS]`, indexed
`[segment_id][plane_kind][dc=0/ac=1]` (`plane_kind`: 0 = luma, 1 = chroma -- `get_dc_quant`/
`get_ac_quant` only ever branch on `plane == 0` vs `!= 0`, so chroma U and V legitimately share
row 1). Built once in `TileDecoder::new_with_prev` from the header/segmentation values (all
fixed for the whole frame) and stored as `TileDecoder::dequant_table`.
`tokens_and_reconstruct` now does a single array lookup (`let [dc_quant, ac_quant] =
self.dequant_table[segment_id as usize][plane_type];`, reusing the `plane_type` already
computed earlier in the function for `coef_probs`) instead of reconstructing a
`SegQIndexOverride` and calling `get_qindex`/`get_dc_quant`/`get_ac_quant` on every transform
block. `get_qindex`/`get_dc_quant`/`get_ac_quant`/`SegQIndexOverride` themselves are untouched
(still unit-tested in `quant.rs`, now called only from the table builder instead of per block).

Removed the now-caller-less `TileDecoder` fields `base_q_idx`/`delta_q_y_dc`/`delta_q_uv_dc`/
`delta_q_uv_ac` (grepped the whole crate first: their only reads were the five lines just
replaced in `tokens_and_reconstruct`; `TileDecoder` is private with two constructors, so no
external code could be relying on the fields either). `self.segmentation` and
`seg_feature_active` stay -- both are still live for `SEG_LVL_SKIP`/`SEG_LVL_REF_FRAME`
elsewhere in the file.

### 4. `src/tile.rs::append_sub8x8_mvs`: `Vec::with_capacity(2)` -> `[Mv; 2]` + len

Mirrors the fixed-array-plus-length-counter pattern `find_mv_refs` already uses via the
existing `add_mv_ref_list(list: &mut [Mv; 2], count: &mut usize, candidate)` helper (`src/mv.rs`,
predates this wave). Reused `add_mv_ref_list` directly for the two dedup-with-cap-2 loops (its
"skip if already at 2" and "skip if equal to slot 0" semantics are exactly what the original
`Vec`-based loops did by hand). The `block == 0` arm does NOT use `add_mv_ref_list`: the
original code pushes `ref_list_mv[0]` and `[1]` unconditionally, with no dedup, so that arm
sets `sub8x8[0]`/`sub8x8[1]` directly to preserve that (using `add_mv_ref_list` there would
silently drop a legitimately-duplicated second entry -- would have been a behavior change).

### 5. `src/predict.rs::predict_intra`: `above_row`/`left_col`/`pred` -> fixed arrays

`size` <= 32 (the 32x32 max transform), so: `above_row` (needs indices `-1..=2*size-1`, i.e.
length `2*size+1` <= 65) -> `[i32; 65]`; `left_col` (length `size` <= 32) -> `[i32; 32]`; `pred`
(length `size*size` <= 1024) -> `[i32; 1024]`. All three are sliced to their exact logical
length (`&mut buf[..len]`) rather than left at full fixed size, so `pred.fill(value)` in the
`DC_PRED` arm and the `left_col.iter().sum()` in the same function still only touch the
logically-valid region, matching the original `Vec` behavior exactly (not just "harmless
because unread" -- actually equivalent).

### 6. `src/predict.rs::block_inter_predict` / `predict_inter`: the hottest site

`block_inter_predict` returned a freshly `Vec`-allocated `intermediate` (horizontal-filter pass)
and `pred` (final result) on every call; `predict_inter` additionally heap-allocated
`preds: [Vec<i32>; 2]` per call. This runs once per sub-8x8 chroma 4x4 block, multiplying the
allocation count the most of any site in the decoder. Changed `block_inter_predict` to take a
caller-provided `pred: &mut [i32]` output slice instead of returning a `Vec`; `predict_inter`'s
`preds` became `[[i32; MAX_BLOCK_DIM * MAX_BLOCK_DIM]; 2]` (a fixed local, not a `TileDecoder`
field -- see Judgment calls), passing `&mut pred_slot[..w*h]` down. `intermediate` inside
`block_inter_predict` became a local `[i32; MAX_INTERMEDIATE_HEIGHT * MAX_BLOCK_DIM]`.

`MAX_BLOCK_DIM = 64`: the largest coding block is 64x64 luma; chroma calls are always <= that
due to subsampling (confirmed both `predict_inter` call sites in `tile.rs` pass
`num4x4w/h * 4`, derived from `NUM_4X4_BLOCKS_WIDE/HIGH_LOOKUP[plane_sz]`, which for the plane's
own block size never exceeds 64).

`MAX_INTERMEDIATE_HEIGHT = 134`: this is the one bound in this wave that needed real derivation
rather than just reading off an existing size, and it disagrees with the task brief's own rough
estimate (`(64+7)`-ish, i.e. assuming the vertical step is always exactly 16 = 1x/unscaled).
Reference-frame scaling is a real, exercised code path here (`RefPlaneView::width/height` come
from the *reference* frame's own decoded dimensions in `Dpb`, which can differ from the current
frame's `frame_width/height` -- VP9 supports decoding at a different resolution than its
references), so `y_step` is not always 16. Spec §8.5.2.3 makes `RefFrameHeight <= 2 *
FrameHeight` a bitstream-conformance requirement; working through the integer arithmetic in
`scale_mv_for_plane` (`y_scale = (ref_height << 14) / frame_height`, `step_y = (16 *
y_scale) >> 14`) shows this caps `y_scale <= 2<<14` exactly (achieved when `ref_height == 2 *
frame_height` exactly, no floor-division loss) and therefore `step_y <= 32`, not just "roughly
16". With `h <= 64` and `step_y <= 32`: `intermediate_height = (((64-1)*32+15)>>4)+8 == 134`.
Used 134 (not the brief's rough estimate) plus `debug_assert!`s on both the block-dimension and
intermediate-height bounds at the top of `block_inter_predict` (zero cost in release, would
fail loudly under the full test suite in debug if the derivation were ever wrong or the spec
assumption violated) rather than trusting the estimate outright.

### Judgment calls

- Local fixed-size stack arrays (not `TileDecoder`-owned persistent scratch fields) for every
  site in this wave, including the two the task suggested could go either way (`tokens`/
  `dequant` in `tile.rs`, `intermediate`/`preds` in `predict.rs`). Reasons: (1) it fully
  satisfies the stated goal (zero heap allocation on these paths) without the cross-field
  aliasing/borrow-splitting the task flagged as a risk (`predict_inter`/`block_inter_predict`
  are free functions in a different module than `TileDecoder`; threading scratch fields through
  would mean widening their already-`#[allow(clippy::too_many_arguments)]` signatures and
  proving disjoint-field-borrow safety at every call site); (2) it keeps each diff local to the
  function being changed, which matters given the bit-exactness bar; (3) the per-call
  zero-initialization of a stack array is far cheaper than the allocator round-trip it replaces
  even where LLVM can't prove full coverage and elide the memset. A `TileDecoder`-owned scratch
  field remains a valid follow-up if profiling on real (larger, non-conformance-vector) content
  ever shows the residual zero-init cost matters.
- Reused the pre-existing `add_mv_ref_list` helper in `append_sub8x8_mvs` (`src/mv.rs`) rather
  than hand-rolling equivalent cap/dedup logic against the new fixed array, since it was already
  imported in `tile.rs` and its semantics are exactly what two of the three push sites needed
  (see item 4) -- less new code, and reuses logic already covered by `mv.rs`'s own unit test.
- Did not touch `SegQIndexOverride` beyond continuing to construct it (now only inside
  `build_dequant_table`) -- per the task's explicit instruction, its dissolution is W5's.

### Verification

`cargo test`: 147/147 pass (no count change from Wave 4a). Conformance: all 5 vector tests
print real `[ok]`-equivalent pass lines against real vector paths (`vp90_2_09_subpixel_00`,
`vp90_2_12_droppable_1`, `vp90_2_15_segkey`, `vp90_2_15_segkey_adpq`,
`vp90_2_16_intra_only`), unchanged bit-exact output. `VP9DEC_FFMPEG="<path-to-ffmpeg>" cargo test --test synthetic_seg_test
synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`: 8/8 `[xdecode] ... OK` lines
(4 synthetic streams x {libvpx-vp9, vp9} ffmpeg decoders). `cargo clippy --all-targets`: same 3
pre-existing warnings as Wave 4a (confirmed identical via `git stash`), no new ones -- in
particular no `needless_range_loop` from the new indexed loops. `rustfmt` run per touched leaf
file (`transform.rs`, `tile.rs`, `predict.rs`; none needed reformatting beyond what the edits
themselves already matched). `git diff --stat`: exactly the 3 expected files (`transform.rs`,
`tile.rs`, `predict.rs`) plus this notes file; no changes to `quant.rs` (the table builder lives
in `tile.rs`, next to its only caller, rather than `quant.rs`, since it's `tile.rs`-specific
plumbing -- `quant.rs` itself needed no changes).

Timing (`cargo clean --release; cargo test --release --test conformance_test`, this machine;
note: this environment's release profile needs `RUST_MIN_STACK` raised for the ThinLTO codegen
worker thread to avoid a stack overflow in `rustc` itself -- reproduced identically at the
pre-Wave-4b baseline via `git stash`, so it's a pre-existing environment quirk unrelated to this
wave's code, not a regression): baseline (5 runs) 2.65/2.65/2.62/2.63/2.64s; after (5 runs)
2.75/2.72/2.64/2.63/2.72s -- within noise, no measurable wall-time delta either way. Debug
profile: similarly within noise (~46-50s across both). Expected and consistent with Wave 4a's
finding: the 5 conformance vectors are small (largest 352x288, few frames), so wall time is
dominated by fixed per-process/per-frame overhead (bool-decoding, I/O, loop filter) rather than
the allocator traffic this wave removes. The qualitative win is the allocation *count*: for a
single 16x16 ADST-ADST transform block alone, the old code made on the order of 64 per-row/
per-column heap allocations inside `idct_permute`/`adst_input_permute`/`adst_output_permute`
(2 per row x 16 rows + 2 per column x 16 columns) plus 3 more (`tokens`, `dequant`, and
`inverse_transform_block`'s own row/column scratch) -- all now zero; every sub-8x8 chroma 4x4
inter-predicted block used to allocate 3 `Vec`s (`intermediate`, `pred` x{1,2 for compound}) --
also now zero. This scales with block/frame count, so the benefit should grow with real-world
(larger, more heavily-populated) content even though it doesn't move the needle on these tiny
conformance clips.

## Wave 5: module boundaries (2026-07-16)

### Scope

Pure structural refactoring -- code motion and signature grouping, zero behavior change.
Split the 2813-line `tile.rs` monolith at its domain seams, unified three duplicated spec
constants/helpers, dissolved two documented seam-structs (`mv.rs`'s "half extraction",
`SegQIndexOverride`), grouped `predict_inter`'s 20 positional args, and added
`BoolDecoder::flag()`. Every diff traces to relocation, signature grouping, or the minimal
visibility bump (`pub(super)`) the relocation itself required -- no arithmetic expression was
touched. Bit-exactness was the gate throughout (full test suite + conformance + ffmpeg
cross-decode, all green before and after).

### 1. `tile.rs` -> `tile/` directory

`tile.rs` stays as the module file (the hub); it declares `mod mode_info; mod mv_pred; mod
ref_ctx; mod residual;`, each backed by `src/tile/<name>.rs` and implemented as one or more
`impl TileDecoder` blocks (methods stay methods -- this is Rust's standard "split a module
across a directory of files" mechanism, not a new abstraction). Final shape:

- `tile.rs` (853 lines, hub): `TileError`, `MiInfo`/`MiGrid` (+ `Default`/inherent impls,
  shared by every submodule and by `loop_filter.rs`), `get_tile_offset`, the `TileDecoder`
  struct/fields, and the traversal `impl` block (`new`/`new_with_prev`, accessors,
  `apply_loop_filter`, `clear_above_context`/`clear_left_context`, `decode_tiles`/
  `decode_tile`/`decode_partition`/`read_partition`/`decode_block`). Keeps the 4 tests that
  exercise `decode_tiles` end-to-end (`get_tile_offset_matches_spec_formula`,
  `single_skip_block_decodes_without_residual_error`,
  `non_skip_block_with_all_zero_tokens_decodes_successfully`, `invalid_tile_size_is_rejected`).
- `tile/mode_info.rs` (1085 lines): `intra_frame_mode_info`/`inter_frame_mode_info` and
  everything they call directly -- segment id (`intra_segment_id`/`inter_segment_id`/
  `get_segment_id`/`seg_feature_active`), `read_skip`/`read_tx_size`, `read_is_inter`/
  `intra_block_mode_info`/`read_ref_frames`, `inter_block_mode_info`/`assign_mv`/`read_mv`/
  `read_mv_component`. Also holds `NeighborRefInfo` (`pub(super)`, constructed here, read by
  `ref_ctx`'s methods) and `interp_filter_ctx` (a judgment call: it's a context-derivation
  function like `ref_ctx`'s, but unlike those it reads `self.mi_grid` rather than being a pure
  function of `NeighborRefInfo`, and its only caller is `inter_block_mode_info` in this same
  file, so it stayed here rather than moving to `ref_ctx.rs`). Only `intra_frame_mode_info`/
  `inter_frame_mode_info` needed `pub(super)` (called from the hub's `decode_block`) --
  everything else this file calls internally stayed plain-private. Moved the segmentation unit
  tests, the `read_is_inter`/`read_ref_frames` `SEG_LVL_REF_FRAME`-forcing tests, and the MV
  round-trip tests (`read_mv`/`read_mv_component`) here, next to their subjects; duplicated the
  small `minimal_header`/`no_segmentation`/`default_compressed_header` test fixtures (also
  needed by the hub's own tests) rather than inventing a shared cross-file test-helpers module
  for ~15 lines of fixture code.
- `tile/ref_ctx.rs` (286 lines): the pure neighbor-context derivation methods --
  `comp_mode_ctx`/`comp_ref_ctx`/`single_ref_p1_ctx`/`single_ref_p2_ctx`, all `pub(super)`
  (called cross-file from `mode_info::read_ref_frames`). No tests of its own: the pre-existing
  test suite never exercised these directly (`read_ref_frames_seg_lvl_ref_frame_returns_...`
  short-circuits past them via `seg_feature_active(SEG_LVL_REF_FRAME)`), so there was nothing to
  move -- not a coverage change made by this wave.
- `tile/mv_pred.rs` (377 lines): `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs` (all
  `pub(super)`, called cross-file from `mode_info::inter_block_mode_info`) and their
  neighbor-scanning helpers (`is_inside`/`get_block_mv`/`if_same_ref_frame_add_mv`/
  `if_diff_ref_frame_add_mv`, all plain-private -- called only within this file). Also fully
  absorbs the former `src/mv.rs` (`Mv`, `ZERO_MV`, `MVREF_NEIGHBOURS`, `MV_BORDER`,
  `MV_PRED_BORDER`, `COMPANDED_MVREF_THRESH`, `MI_SIZE`, `clamp_mv_row`/`clamp_mv_col`/
  `add_mv_ref_list`/`scale_mv`/`use_mv_hp`, plus its 4 unit tests, moved verbatim): `mv.rs`'s own
  module doc said its bodies stayed in `tile.rs` only for borrow convenience with
  `TileDecoder`/`MiGrid`, and now that the real methods live in this same submodule, keeping the
  pure helpers in a separate top-level module served no purpose. `Mv` is `pub` and re-exported
  at the hub (`pub use mv_pred::Mv;`, so `crate::tile::Mv`) since `predict.rs` is the one
  external (non-`tile`) consumer; `ZERO_MV`/`use_mv_hp` are `pub(super)` (also used from
  `mode_info.rs`); everything else former-`mv.rs` stayed plain-private (only used within this
  file after the merge).
- `tile/residual.rs` (473 lines): `residual` (`pub(super)`, called from the hub's
  `decode_block`) and everything it calls directly -- `get_uv_tx_size`/`get_plane_block_size`,
  `compute_tx_type`/`tx_sz_to_scan_size`, `tokens_and_reconstruct`/`read_coef` -- plus the W4b
  per-frame dequant table builder `build_dequant_table` (a free fn, `pub(super)`, called from
  the hub's `TileDecoder::new_with_prev`; it moved here rather than staying in the hub because
  its only reason to exist is feeding this file's dequantization step).

Mechanically, cross-file calls needed `pub(super)` (Rust privacy is subtree-based: a
plain-private item is visible in its defining module and that module's descendants, so a
private method defined inside `tile/mode_info.rs` is *not* visible from `tile.rs` itself or from
`tile/mv_pred.rs` -- only from `mode_info.rs` and its own descendants -- while `pub(super)`
makes it visible throughout `tile`'s whole subtree). Struct *fields* needed no visibility
change at all: `TileDecoder`'s fields are declared plain-private in the hub, and every
submodule, being a descendant of `tile`, can already read/write `self.<field>` from an `impl
TileDecoder` block physically located in a different file.

### 2. `src/common.rs`: unify 3 duplicated spec constants/helpers

New `#[doc(hidden)] pub mod common` (same convention as every other internal module, per
`lib.rs`) holding the single canonical definition of:

- `MAX_SEGMENTS: usize = 8` -- was `pub` in `header.rs` and privately redefined in
  `loop_filter.rs`. `header.rs` now does `pub use crate::common::MAX_SEGMENTS;` at the same
  spot, so `header::MAX_SEGMENTS` (imported by `tile.rs` and by `tests/common/encoder.rs`)
  keeps working unchanged; `loop_filter.rs` imports it straight from `common`.
- `INTRA_FRAME: u8 = 0` -- was `pub` in `prob_tables.rs` (re-exported by `header.rs`) and
  privately redefined as a `usize` in `loop_filter.rs`. `prob_tables.rs` now does `pub use
  crate::common::INTRA_FRAME;` at the same spot (keeps `prob_tables::INTRA_FRAME` and
  `header`'s further re-export working); `loop_filter.rs` imports it straight from `common` and
  its 3 `usize`-indexing call sites gained an explicit `as usize` (the type is `u8` now, unified
  with every other `ref_frame`-typed value in the codebase, rather than a `loop_filter`-local
  `usize`).
- `get_uv_tx_size(mi_size, tx_size, subsampling_x, subsampling_y) -> u8` -- was duplicated
  verbatim as a `TileDecoder` method (deriving `subsampling_x`/`subsampling_y` from `self` via
  `get_plane_block_size`) and as a `loop_filter.rs` free function (taking them as params
  directly). Unified on the free-function form (`loop_filter.rs`'s shape, since it has no
  `self` to borrow from); `TileDecoder::get_uv_tx_size` (`tile/residual.rs`) is now a one-line
  wrapper passing `self.subsampling_x`/`self.subsampling_y` through, so its call site
  (`self.get_uv_tx_size(...)` in `residual()`) needed no change at all.

### 3. `SegQIndexOverride` dissolved (`src/quant.rs`)

`get_qindex(base_q_idx, seg: Option<SegQIndexOverride>) -> u8` -> `get_qindex(base_q_idx,
segmentation: &header::SegmentationParams, segment_id: usize) -> u8`. The struct existed only
to keep `quant.rs` decoupled from `header::SegmentationParams`; W2b's `common` precedent made
that reasoning moot (judgment call, per the task brief), and passing the params straight
through lets `get_qindex` mirror the spec's `seg_feature_active( SEG_LVL_ALT_Q )` branch
directly instead of requiring the caller to pre-derive an `Option`. The caller
(`build_dequant_table`, now in `tile/residual.rs`) shrank from a 9-line `Option` construction
to a single `get_qindex(base_q_idx, segmentation, segment_id)` call. Updated `quant.rs`'s 3
affected unit tests to build a `SegmentationParams` (via new local test helpers
`no_segmentation`/`seg_lvl_alt_q`) instead of a `SegQIndexOverride`; no other file referenced
the struct (confirmed via search before deleting it).

### 4. `predict_inter` parameter grouping (`src/predict.rs`)

New `pub struct InterPredictParams<'a>` groups the 14 of `predict_inter`'s 20 positional
parameters that are identical across every call one coding block's `residual()` makes (one
call per plane, further split into 4x4 sub-blocks when `MiSize < BLOCK_8X8`): `ref_frame`,
`block_mvs`, `interp_filter`, `mi_row`/`mi_col`/`mi_size`/`mi_rows`/`mi_cols`,
`subsampling_x`/`subsampling_y`, `frame_width`/`frame_height`, `bit_depth`, `refs`.
`predict_inter` itself is now `(dst, plane, x, y, w, h, block_idx, p: &InterPredictParams)` --
8 params (still `#[allow(clippy::too_many_arguments)]`, one over the 7 threshold, down from
20). Call sites in `tile/residual.rs` build one `InterPredictParams` before the
sub8x8-loop-vs-full-block `if`/`else` (identical either way) instead of repeating the same
14 values twice.

Applied the same struct to `predict_inter`'s 3 private helpers, since all 3 already draw
their invariant inputs from this exact same field set:
`select_mv(plane, ref_list, block_idx, p)` (was 7 params -- not over the clippy threshold, but
made consistent with its 2 siblings since all its non-`p` inputs are per-call);
`clamp_mv_for_plane(plane, mv, p)` (was 9, had `#[allow(too_many_arguments)]` -- now 3, allow
removed); `scale_mv_for_plane(plane, x, y, clamped_mv, ref_width, ref_height, p)` (was 10, had
the allow -- now 7, right at the threshold, allow removed; `ref_width`/`ref_height` stayed
positional since they're the specific reference's own size, read from `p.refs[ref_list]` at
the call site, not frame-invariant).

Left `block_inter_predict` (10 params) and `predict_intra` (11 params) as-is: `predict_intra`
is a different pipeline (intra, not inter) with no comparable invariant/per-call split --
every one of its params genuinely varies call to call; `block_inter_predict` only overlaps
`InterPredictParams` in 2 of its 10 fields (`interp_filter`, `bit_depth`), so threading the
whole struct through for a 10 -> 9 reduction wasn't judged worth the added coupling. Also left
the `tile/residual.rs` / `tile/mode_info.rs` / `tile/mv_pred.rs` `#[allow(too_many_arguments)]`
sites untouched (`residual`, `compute_tx_type`, `tokens_and_reconstruct`, `inter_block_mode_info`,
`append_sub8x8_mvs`): each one's arguments are already a mix of genuinely-independent per-call
values (`row`/`col`/`plane`/`start_x`/`start_y`/etc.) with no natural invariant subset to
extract without inventing a struct that would just re-list most of `TileDecoder`'s own fields.

### 5. `prob_tables.rs` domain split

Moved `SUBPEL_FILTERS` + the 5 motion-compensation constants (`REF_SCALE_SHIFT`/`SUBPEL_BITS`/
`SUBPEL_SHIFTS`/`SUBPEL_MASK`/`INTERP_EXTEND`) to new `src/subpel.rs` (98 lines); moved
`MV_REF_BLOCKS`/`MODE_2_COUNTER`/`COUNTER_TO_CONTEXT`/`IDX_N_COLUMN_TO_SUBBLOCK` to new
`src/mv_ref_tables.rs` (151 lines) rather than into `tile/mv_pred.rs` -- kept as a standalone
top-level module (mirroring `subpel.rs`) since `prob_tables.rs` itself is a flat table module
with no submodule structure of its own, so a same-shaped sibling was the smaller conceptual
jump than reaching into `tile`'s private subtree. Both are `#[doc(hidden)] pub mod`, same
convention. `predict.rs` now imports the subpel items from `crate::subpel`; `tile/mv_pred.rs`
imports the MV-ref tables from `crate::mv_ref_tables`. Neither set of items had any other
importer (`tests/common/encoder.rs` imports neither), so this was a clean move with no
re-export needed (unlike `common`'s `MAX_SEGMENTS`/`INTRA_FRAME`, which needed re-exports
because outside code already depended on their old import paths). Two of the moved doc
comments had their `[`NEARESTMV`]`/`[`NEWMV`]` intra-doc-link brackets flattened to plain code
spans in `mv_ref_tables.rs` (they relied on those constants being in the same file as the
comment in `prob_tables.rs`; re-importing them just to keep a doc link alive wasn't worth it).
`prob_tables.rs` (2024 lines, down from 2270) now holds only trees/probabilities/block-geometry,
matching its own doc comment's description of its scope.

### 6. `BoolDecoder::flag()`

Added `pub fn flag(&mut self) -> bool` (`read_literal(1) == 1`) to `bool_coder.rs`. Replaced the
6 genuine boolean-comparison call sites in `compressed_header.rs`
(`decode_term_subexp`'s 3 `read_literal(1) == 0` checks -> `!r.flag()`; `read_coef_probs`'s
`update_probs`, `frame_reference_mode`'s `non_single_reference`/`reference_select`, all
`read_literal(1) == 1` -> `r.flag()`). Left 2 other `read_literal(1)` call sites untouched
(`decode_term_subexp`'s final `let bit = r.read_literal(1);` and `read_tx_mode`'s `let
tx_mode_select = r.read_literal(1) as u8;`): both use the result as a `u32`/`u8` in arithmetic
(`(v << 1) - 1 + bit`, `tx_mode += tx_mode_select`), not as a boolean comparison, so they don't
match the `read_literal(1) == 1`/`== 0` pattern the task described.

### Judgment calls (summary; see inline call-outs above for the reasoning)

- `interp_filter_ctx` stayed in `mode_info.rs` rather than moving to `ref_ctx.rs` (reads
  `self.mi_grid`, not a pure function of `NeighborRefInfo` like its 4 `ref_ctx` neighbors).
  `NeighborRefInfo` itself stayed in `mode_info.rs` (constructed there) with `pub(super)`
  fields, rather than moving to `ref_ctx.rs` (consumed there) or a new standalone module.
- Duplicated small test fixtures (`minimal_header`/`no_segmentation`/`default_compressed_header`)
  across `tile.rs` and `tile/mode_info.rs` rather than adding a shared test-helpers module.
- `get_qindex`'s replacement signature takes `&header::SegmentationParams` directly (the
  brief's fallback option) rather than inventing a new minimal shared type, since `common.rs`
  already removed the original decoupling rationale entirely.
- `mv_ref_tables.rs` kept as a standalone top-level module rather than folded into
  `tile/mv_pred.rs` (see §5).
- `block_inter_predict`/`predict_intra` left with their existing `#[allow(too_many_arguments)]`
  (see §4) -- the task explicitly said not to force grouping without a natural split.

### Verification

`cargo build`: clean, no warnings. `cargo test`: 147/147 across all 6 binaries, same total as
the Wave 4b baseline (0 shift) -- per-binary: `vp9dec` (lib) 119 (identical to the pre-wave
baseline, confirmed via `git stash`: the tests that moved -- `mv.rs`'s 4 and `tile.rs`'s 11
relocated to `tile::mv_pred`/`tile::mode_info` -- were already counted under the same
`unittests src\lib.rs` binary before this wave, so the split moved them between source files
and module paths, not between test binaries), `api_test` 6, `conformance_test` 12,
`synthetic_seg_test` 6, `decode_to_png` example 4, doc-tests 0. Conformance: all 5 vectors
print real `[ok]` MD5-match
lines (`vp90-2-16-intra-only` 7 frames, `vp90-2-15-segkey` 1 frame, `vp90-2-09-subpixel-00` 20
frames, `vp90-2-12-droppable_1` 99 frames, `vp90-2-15-segkey_adpq` 150 frames), unchanged from
baseline. `VP9DEC_FFMPEG="<path-to-ffmpeg>" cargo
test --test synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`:
8/8 `[xdecode] ... OK` lines (4 synthetic streams x {libvpx-vp9, vp9} ffmpeg decoders).
`cargo clippy --all-targets`: same 3 pre-existing baseline warnings as Wave 4b (confirmed
identical via `git stash`; `header.rs` large_enum_variant, `superframe.rs` identity_op,
`lib.rs` field_reassign_with_default), no new ones. `RUSTDOCFLAGS="-D
rustdoc::broken_intra_doc_links" cargo doc --no-deps`: no broken links; the pre-existing 3
`private_intra_doc_links` warnings (confirmed identical via `git stash`: `TileDecoder::residual`/
`tokens_and_reconstruct`/`crate::Decoder::prev_mi_grid`) are unchanged -- the module doc
comment added to `tile.rs` initially introduced 4 more (linking to the new private submodules
by name) but these were flattened to plain code spans (no intra-doc link) once confirmed
unnecessary. `rustfmt --check` run per new file only (`common.rs`/`subpel.rs`/
`mv_ref_tables.rs`/`tile/mode_info.rs`/`tile/ref_ctx.rs`/`tile/mv_pred.rs`/`tile/residual.rs`);
applied where it flagged anything (import-line wrapping and 2 function signatures in
`tile/mv_pred.rs` that got shorter after dropping their old `pub` keyword but still needed
multi-line wrapping) -- `tile.rs` itself (pre-existing, not new) was never run through rustfmt.
`git diff --stat`: `tile.rs` 2813 -> 853 lines; new `tile/mode_info.rs` (1085) /
`tile/ref_ctx.rs` (286) / `tile/mv_pred.rs` (377) / `tile/residual.rs` (473); `mv.rs` deleted
(125 lines); new `common.rs` (25) / `subpel.rs` (98) / `mv_ref_tables.rs` (151); `quant.rs`,
`predict.rs`, `loop_filter.rs`, `header.rs`, `compressed_header.rs`, `bool_coder.rs`, `lib.rs`
(mod decls) all touched as described above; no changes to any file under `tests/`.

## Wave 6: docs + tooling (2026-07-16)

### Scope

Final wave of the design-debt plan: brought README.md, this notes file, and the vector-fetch
tooling in line with the architecture Waves 1-5 actually produced (public API shape, module
layout, test-file names), none of which README.md had caught up to (it still described
`decode_keyframe()`, the pre-Wave-2b `decode_frame` signature, and the pre-Wave-3 test files).
No behavior change: docs and shell/PowerShell scripts only.

### README.md restructure

Replaced the M1 -> M3-second-half changelog narrative with five present-tense sections:
Purpose (updated: `test-support` is currently the *only* dev-dependency, spelled out by name
rather than left as a hypothetical "may use dev-dependencies where useful"); Current
architecture (the public API surface -- `Decoder`/`DecodedFrame`/`Frame`/`FrameDecodeInfo`/
`DecodeError`/`ivf` -- and a module map grouped by pipeline stage, one paragraph per stage,
written from the current module docs and `pub`/`#[doc(hidden)]` structure rather than
transcribed from memory of the old text); a Status/limitations table split into "proven
against an external oracle" vs. "known limits" (the reference-scaling caveat and the 8-bit
limit are now a table, not prose buried in a milestone section); Tests & verification (what
`cargo test` actually runs across its 6 binaries, per-file coverage, and the ffmpeg/
`VP9DEC_DUMP_DIR` invocation forms in both bash and PowerShell); and a two-line History
section pointing at the two files below instead of holding the narrative itself.

Every concrete number in the new README (147 tests across 6 binaries with the per-binary
breakdown 119/6/12/6/4/0, the 5 conformance vectors' exact frame counts, the 8 `[xdecode]`
lines from the ffmpeg cross-decode, the `[ok]`/`[coverage]` lines) was reproduced by actually
running `cargo test`, `cargo test --test conformance_test -- --nocapture`, and
`VP9DEC_FFMPEG=... cargo test --test synthetic_seg_test
synthetic_streams_cross_decode_against_ffmpeg -- --nocapture` in this session rather than
copied from an earlier wave's notes entry (those numbers can drift wave to wave, so this
wave's own re-observation is what backs the README claims, not an assumption of continuity).

### docs/history.md (new)

Moved the M1/M2/M2b/M3-first-half/M3-second-half narrative sections out of README.md
verbatim (light copy-editing only for flow, e.g. tense fixes now that they're explicitly
framed as historical), preserving the original prose including judgment calls and the
debugging-trail section. Added bracketed `[...]` pointers at the specific paragraphs whose
claims were later superseded (the `decode_keyframe()`/`Decoder::decode_frame()` signatures
across three different shapes, `src/mv.rs`'s later absorption into `tile/mv_pred.rs`,
`src/md5.rs`'s relocation to `tests/common/md5.rs`, the `more_coefs` counting bug, the
`SUBPEL_FILTERS` move to `subpel.rs`, and the MV-ref-tables move to `mv_ref_tables.rs`) —
per the task's instruction, these are pointers to the correcting notes entry, not rewrites of
the historical text itself. Also moved the PNG-dump usage notes and the original
`curl`/`Invoke-WebRequest` vector-download commands out of README (both superseded by
`scripts/fetch-vectors.{sh,ps1}` below), keeping them as a "how it used to be done" appendix.

### scripts/fetch-vectors.sh + scripts/fetch-vectors.ps1 + scripts/vectors.txt (new)

Manifest-driven (`scripts/vectors.txt`: `<name> <kind>` per line, `kind` = `ivf`|`webm`,
currently the same 5 vectors `tests/conformance_test.rs`/`tests/synthetic_seg_test.rs`
already reference: `vp90-2-12-droppable_1`/`vp90-2-09-subpixel-00` as `ivf`,
`vp90-2-15-segkey`/`vp90-2-15-segkey_adpq`/`vp90-2-16-intra-only` as `webm`). Both scripts
download `<name>.<ext>` + `<name>.<ext>.md5` from
`storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/` into `tests/vectors/`,
skipping any file already present; for `webm` entries, additionally remux via `cargo run
--example webm_to_ivf -- tests/vectors/<name>.webm tests/vectors/<name>.ivf` (skipped if the
`.ivf` already exists) and copy `<name>.webm.md5` to `<name>.ivf.md5` (skipped likewise). The
`.sh` uses `curl -fSL` under `set -euo pipefail` (an HTTP error or `webm_to_ivf` non-zero
exit aborts the script immediately, satisfying "fail loudly"); the `.ps1` uses
`Invoke-WebRequest` under `$ErrorActionPreference = "Stop"` plus an explicit `$LASTEXITCODE`
check after the `cargo run` call (`Invoke-WebRequest` itself already throws a terminating
`WebException` on a non-2xx response).

Verified against the real (non-empty) `tests/vectors/` in this repo, without deleting
anything: both scripts printed 16 `[skip] ... already present` lines (2 files each for the
2 `ivf`-kind entries + 4 files each for the 3 `webm`-kind entries = 16) and exited 0 —
`bash scripts/fetch-vectors.sh` and `pwsh -File scripts/fetch-vectors.ps1` (`pwsh` 7 is
present on this machine) both confirmed. The download path itself (not just the skip
branch) was additionally verified for real, outside the scripts: `curl -fSL` against
`.../vp90-2-09-subpixel-00.ivf.md5` (the same URL pattern the scripts build) into a scratch
directory succeeded and returned the expected `md5sum`-format content, without touching
`tests/vectors/`. A full from-empty run of either script (which would additionally exercise
the `webm_to_ivf` remux branch) was not performed, since that requires deleting the existing
`.ivf`/`.ivf.md5` files the conformance suite depends on — out of scope per the task's "do
NOT delete vectors to test the download path" constraint. The remux branch's underlying
command (`cargo run --example webm_to_ivf -- <in> <out>`) is unchanged from what
`examples/webm_to_ivf.rs` already does and is already covered by that example's own history
(see docs/history.md and the "WebM remux" notes entry above) — this wave only wraps it in a
skip-if-present shell/PowerShell loop, so re-deriving fresh confidence in `webm_to_ivf.rs`
itself was judged unnecessary.

### One permitted src/ comment fix

`src/compressed_header.rs`'s `FrameContextStore` doc comment referenced
`parse_compressed_header_ex` — a name deleted in Wave 2a when `parse_compressed_header`/`_ex`
were collapsed into one 3-argument `parse_compressed_header` (see that wave's entry above).
Reworded the one line to say `parse_compressed_header` (confirmed via `grep` that
`starting_probs` is still the correct parameter name on the current function signature). No
other line in that comment, or anywhere else in `src/`, needed changing — grepped the whole
of `src/` for `decode_keyframe`/`last_frame_info`/`SegQIndexOverride`/
`parse_compressed_header_ex`/`src/mv.rs`/`src/md5.rs` first; the only other hits
(`src/quant.rs`'s `SegQIndexOverride` mention, `src/tile.rs`'s and
`src/tile/mv_pred.rs`'s `src/mv.rs` mentions) already correctly describe those as no-longer-
existing (past tense / "former standalone"), so were left alone.

### Consistency sweep

Grepped README.md (before this wave's rewrite) for the deleted/renamed names the task
listed; after the rewrite, every one of those strings appears only inside
`docs/history.md` (the historical record, where they're expected) or not at all in
README.md itself. See the parent task's report for the exact before/after counts.

### Verification

`cargo test`: 147 passed, 0 failed, across the same 6 binaries as the Wave 5 baseline (119
lib unit + 6 `api_test` + 12 `conformance_test` + 6 `synthetic_seg_test` + 4
`decode_to_png` example + 0 doc-tests) — unchanged, confirming this wave touched no
decode-affecting code (the one `src/` edit is a comment-only line). Conformance
(`cargo test --test conformance_test -- --nocapture`): all 5 vectors' `[ok]` lines
unchanged (`vp90-2-12-droppable_1` 99/99, `vp90-2-09-subpixel-00` 20/20,
`vp90-2-15-segkey` 1/1, `vp90-2-15-segkey_adpq` 150/150, `vp90-2-16-intra-only` 7/7).
`VP9DEC_FFMPEG="<path-to-ffmpeg>" cargo test
--test synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`:
8/8 `[xdecode] ... OK` lines (4 synthetic scenarios x {libvpx-vp9, vp9}), unchanged.
`git diff --stat`: `README.md` (rewritten), `docs/history.md` (new), `docs/
implementation-notes.md` (this section + the Current state index), `scripts/vectors.txt` /
`scripts/fetch-vectors.sh` / `scripts/fetch-vectors.ps1` (new), and the single-line
`src/compressed_header.rs` comment fix — nothing else in `src/` or `tests/`.

## Design-debt redesign: closing summary (2026-07-16)

The redesign that began with hardening fixes discovered while chasing conformance (full
segmentation support, the `intra_only` frame-context-reset bug, the conformance-coverage
instrumentation, the synthetic round-trip + ffmpeg cross-decode harness for the three
`SEG_LVL_*` features with no official vector, the superframe-splitting fix, and the
coefficient EOB-branch counting fix — the dated entries from 2026-07-12 through 2026-07-14
above) continued into six explicitly-labeled "design-debt waves" addressing structural debt
that had accumulated across the milestone-driven M1-M3 development: Wave 1 (stale comments,
dead constants, shared `test_support`/`write_ivf` infra), Wave 2a (internal signature
redesign — `PersistentState`, honest `Option<ColorConfig>`), Wave 2b (public API redesign —
`decode_frame` -> `Vec<DecodedFrame>`, `decode_keyframe`/`last_frame_info` deleted,
`#[doc(hidden)]` narrowing), Wave 3 (test-layer consolidation — 6 integration test files
down to 3 plus `tests/common/`), Wave 4a (`Arc`-sharing to eliminate per-frame deep clones),
Wave 4b (hot-path allocations removed, per-frame dequant table), Wave 5 (module boundary
refactor — `tile.rs` split into `tile/{mode_info,ref_ctx,mv_pred,residual}.rs`, `common.rs`/
`subpel.rs`/`mv_ref_tables.rs` extracted, `mv.rs`/`SegQIndexOverride` dissolved), and this
Wave 6 (docs + tooling).

Commits (oldest first; Wave 6 is docs/tooling-only and, per this wave's task constraints,
was not committed as part of this work): `dd5f875` (W1), `47db20d` (W2a), `a848a00` (W2b),
`22fc88e` (W3), `a6b8c87` (W4a), `928b605` (W4b), `5c5054d` (W5).

End state, observed in this session: `cargo test` passes 147/147 across 6 binaries; 5
official libvpx conformance vectors are bit-exact against their `.ivf.md5` on every
displayed frame (`vp90-2-12-droppable_1` 99/99, `vp90-2-09-subpixel-00` 20/20,
`vp90-2-15-segkey` 1/1, `vp90-2-15-segkey_adpq` 150/150, `vp90-2-16-intra-only` 7/7,
including segmentation's `SEG_LVL_ALT_Q` and `intra_only`/`reset_frame_context`/superframe
splitting); and the three `SEG_LVL_*` features with no official vector
(`SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP`) are confirmed byte-identical against
two independent third-party VP9 decoders (ffmpeg's `libvpx-vp9` and its native `vp9`) — 8
`[xdecode]` checks (4 synthetic scenarios x 2 decoders), an 8-way cross-decode. `src/`
itself carries zero external dependencies throughout; the one dev-dependency
(`test-support`, a self-reference) never affects a plain `cargo build`. M4 (the full
official libvpx vector sweep beyond these 5) remains open — see README.md's Status/
limitations table and `scripts/fetch-vectors.{sh,ps1}` for extending vector coverage.

## M4 wave 1: full official-vector sweep infrastructure + first honest triage (2026-07-16)

Infrastructure-only wave: built the machinery to run every official libvpx `vp90-2-*`
conformance vector through the decoder and categorized the failures. **No `src/` changes were
made in this wave** (the decoder bugs surfaced below are reported, not fixed — that is a later
wave's job). Only `scripts/`, `tests/`, and this notes file changed.

### Infrastructure added

- **`scripts/vectors.txt`**: expanded from the 5 curated entries to the full official set,
  330 entries (`<name> <kind>` format, unchanged schema). Derived mechanically from
  libvpx's `test/test-data.sha1` list (359 `vp90-2-*` filenames). Excluded: the 12
  `vp90-2-tos_*`/`vp90-2-sintel_*` movie clips (full-length, deferred to a later phase per
  the wave scope) and the 17 `*.res` sidecars (those are libvpx's expected-frame-count
  fixtures for its own corrupted-stream tests, not video containers, so they don't fit the
  `ivf`/`webm` kind schema). 307 `webm`-kind + 23 `ivf`-kind = 330. The 5 pre-existing
  curated entries are preserved in the regenerated list.
- **`scripts/fetch-vectors.sh` / `.ps1`**: reworked from fail-fast (`set -euo pipefail` /
  `$ErrorActionPreference=Stop`) to **continue-on-error with an end-of-run summary**. At 330
  entries some upstream files legitimately 404 and some `.webm` might not remux; aborting on
  the first one would defeat the sweep. Each download/remux failure is recorded and skipped;
  the summary prints counts + every failing name. Also added a one-time
  `cargo build --example webm_to_ivf` before the loop so the ~300 remux invocations reuse the
  built binary.
- **`tests/sweep_test.rs`**: new `#[test] #[ignore] fn official_vector_sweep()`. Scans
  `tests/vectors/*.ivf` that have a matching `.ivf.md5`, decodes ALL IVF chunks per vector
  through one `Decoder`, MD5s each displayed `DecodedFrame`'s I420 bytes (via
  `tests/common/md5`) against the `.ivf.md5` lines. Every failure mode is caught per-vector
  so one bad vector never aborts the sweep: a returned `Err` becomes `error@frame N`, an MD5
  divergence `md5-mismatch@frame N`, a frame-count divergence `md5-count-mismatch`, and a
  panic is caught via `std::panic::catch_unwind` (with the default panic hook suppressed for
  the sweep's duration so a decoder-internal panic doesn't spew backtraces over the report).
  Emits one `[PASS]`/`[FAIL <reason>]` line per vector plus a summary block, and writes the
  same report to `target/sweep-report.txt` for the reviewer. The test asserts all-pass at the
  end, so it **fails today** — that is expected and why it's `#[ignore]`d; the normal
  `cargo test` suite stays green (147/147). Run it with:
  `cargo test --release --test sweep_test official_vector_sweep -- --ignored --nocapture`
  (release for speed; `RUST_MIN_STACK=16777216` was set on the build to avoid the documented
  rustc-side ThinLTO worker-thread stack overflow, an environment quirk noted in earlier
  waves — it did not actually bite this time but the env var was kept as insurance).

### Fetch / remux stats (one full run, this machine)

- downloads ok: **605**, downloads failed (404): **45**, remux ok: **301**, remux failed:
  **0**. The pure-std WebM remuxer (`examples/webm_to_ivf.rs`) handled **every** downloaded
  `.webm` without a single lacing/structure rejection — a good result for the from-scratch
  EBML reader.
- The 45 download 404s break down as: 16 `ivf`-kind "invalid/resilience" entries whose whole
  `.ivf` isn't hosted (the `*.webm.ivf.sNNNNN_r01-05_b6-*.ivf` / `*.ivf.kf_65527x61446.ivf`
  family — libvpx generates these locally for its own decode-of-corrupt tests, they aren't in
  the storage bucket; 32 files = 16 `.ivf` + 16 `.ivf.md5`), 3 fully-absent `webm` vectors
  (`vp90-2-07-frame_parallel-2`, `-3`, `vp90-2-08-tile_1x4_frame_parallel_all_key`; 6 files),
  and 7 `bbb` vectors that ship a `.webm` but **no** `.webm.md5` upstream (7 files). The bbb
  ones still remuxed to `.ivf`, but with no MD5 the sweep can't check them, so they're
  skipped.
- Net on disk: 311 `.ivf` produced (330 − 16 unhosted `.ivf` − 3 absent `webm`), of which
  **304 have a matching `.ivf.md5`** and are therefore swept; the 7 bbb `.ivf` are the
  no-MD5 skips. tests/vectors/ ≈ 1.5 GB.

### Sweep result: 275 / 304 pass (90.5%)

`total 304 / pass 275 / fail 29`, and — notably — **every one of the 29 failures is a plain
`md5-mismatch`: zero decode-errors, zero panics, zero frame-count mismatches.** So every
in-scope bitstream parses and decodes to completion end-to-end (container, superframe split,
all header layers, tile/token decode, DPB, loop filter); only some reconstructed pixels
differ. Full per-vector report: `target/sweep-report.txt`.

### Triage (5 categories, 29 vectors) — hypotheses, NOT fixes

Quick evidence gathered by reading each failing vector's uncompressed header (`base_q_idx` /
`lossless` / dims) via a throwaway probe example (since deleted; not committed). No deep
debugging.

| # | Category (count) | Example vectors | Evidence | One-line hypothesis |
|---|---|---|---|---|
| A | **Low-QP / lossless keyframe recon (9)** | `quantizer-00`..`-07` (8), `13-largescaling` | Crisp QP threshold: `quantizer-00..07` = `base_q_idx` 0,4,…,28 all FAIL @0; `quantizer-08..63` = `base_q_idx`≥32 all PASS. `quantizer-00` and `largescaling` are exactly `lossless` (`base_q_idx==0`). | Inverse-transform intermediate precision/clamping (or high-magnitude dequant) that only trips once coefficients get large enough at low QP; `quantizer-00`/`largescaling` additionally hit the exact-lossless 4×4-WHT path. Not *purely* a lossless bug (only `-00` is lossless, yet `-01..-07` fail). |
| B | **Small odd frame sizes (10)** | `02-size-10x08`, `-10x10`, `-10x32`, `-08x34`, `-08x66`, `-16x66`, `-18x10`, `-34x10`, `-66x08`, `-66x10` | 10 of 71 `02-size` fail (mostly @0 keyframe). No divisibility rule: `w=10` fails at `h∈{8,10,32}` but passes at `h∈{16,18,34,64,66}`; `32` fails yet `34` passes. QP-independent (`08x32` q=1 passes; `08x34` q=93 fails). All 65 `03-size` (196–226) and 3 `11-size` (351/352) PASS. | Edge-block / partial-superblock intra-reconstruction (or boundary pixel extension) bug specific to the tiny (≤66 px) regime; content/partition-specific, not a clean dimension-alignment error. |
| C | **Reference scaling / resize / SVC (4)** | `18-resize` @2, `22-svc_1280x720_3` @0, `14-resize-10frames-fp-tiles-1-2-4-8` @40, `14-…-8-4-2-1` @31 | Basic scaled MC works: `05-resize`, all 16 `21-resize_inter_*`, and 34 of 36 `14-resize-*` PASS. The 2 failing `14-*` are exactly the ones cycling **four** tile configs. `svc_3` (3 spatial layers) fails where `svc_1` (1 layer) passes; its base layer is 320×180. | Reference-frame scaling for specific scale ratios / resize×multi-tile-config transitions / multi-spatial-layer SVC diverges, while single-ratio scaled MC is correct. |
| D | **Mid-stream inter divergence (5)** | `07-frame_parallel-1` @10, `19-skip` @6, `19-skip-01` @8, `20-big_superframe-01` @3, `20-big_superframe-02` @8 | All decode many frames correctly, then one diverges. Siblings pass: `frame_parallel` (base), `skip-02`. | A specific inter-frame feature / probability-adaptation edge case triggered only by certain frames; core inter-prediction is correct (proven by the many matching frames before the break). Not yet localized. |
| E | **show_existing_frame first output (1)** | `10-show-existing-frame` @0 | 2 of 3 show-existing vectors PASS (`10-show-existing-frame2`, `17-show-existing-frame`). | The `show_existing_frame` mechanism itself works; this stream's first *output* diverges, most likely because the hidden frame it references decoded slightly wrong upstream — a symptom of another category, not of show-existing. |

Priority order for a future fix wave, by evidence sharpness: **A** (crispest boundary, likely
a single transform/clamp site) → **B** (well-bounded to the tiny-size regime) → **C**
(scaling, well-isolated) → **D**/**E** (need per-frame localization first).

### Verification

- `cargo test` (normal suite): **147 passed, 0 failed** across 6 binaries (119 lib + 6
  `api_test` + 12 `conformance_test` + 6 `synthetic_seg_test` + 4 `decode_to_png`), unchanged
  from the pre-wave baseline. The new `sweep_test` binary reports `0 passed; 1 ignored` — the
  sweep does not run under a plain `cargo test`, so the suite stays green.
- The 5 curated conformance vectors and their named tests are untouched and still pass
  (they're a subset of the 275 the sweep also passes).
- `git diff`/`status` touches only `scripts/vectors.txt`, `scripts/fetch-vectors.sh`,
  `scripts/fetch-vectors.ps1`, and the new `tests/sweep_test.rs`, plus this notes section.
  **Zero `src/` changes.** (`tests/vectors/*` and `target/` are gitignored, so the downloaded
  vectors and `target/sweep-report.txt` don't appear in the diff.)
