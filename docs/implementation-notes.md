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
