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

| Path | Official E2E vector | Reference-derived vector | Unit test only |
| --- | --- | --- | --- |
| Segmentation seg-id decode (`intra_segment_id`/`inter_segment_id`/`get_segment_id`) | ✓ `vp90-2-15-segkey` -- `intra_segment_id` only (1-frame vector, no inter frame, so `inter_segment_id`/`get_segment_id` temporal prediction remain unproven E2E) | TODO | yes (`src/tile.rs`) |
| `SEG_LVL_ALT_Q` | ✓ **`vp90-2-15-segkey_adpq`** -- all 150 output frames bit-exact (coverage line reports `SEG_LVL_ALT_Q=true`) | TODO | yes (`src/quant.rs`, `src/header.rs`) |
| `SEG_LVL_ALT_L` | none (no official or readily-encodable vector) | TODO | yes (`src/loop_filter.rs`) |
| `SEG_LVL_REF_FRAME` | none (no official or readily-encodable vector) | TODO | yes (`src/tile.rs`) |
| `SEG_LVL_SKIP` | none (no official or readily-encodable vector) | TODO | yes (`src/tile.rs`) |
| `intra_only` frame | ✓ **`vp90-2-16-intra-only`** -- all 7 output frames bit-exact; the 3 hidden `intra_only` priming frames are verified via `show_existing_frame` round-trip and the 4 inter frames referencing them decode correctly (coef EOB-count fix below) | TODO | yes (`src/header.rs`) |
| `reset_frame_context < 3` | ✓ **`vp90-2-16-intra-only`** -- exercises `reset_frame_context == 2` on all 3 of its `intra_only` frames (coverage line reports `reset_frame_context values seen = {0, 2}`) | TODO | yes (`src/lib.rs`: `frame_context_reset_*`) |

Final state (as of the superframe-splitting-API and EOB-count fixes below): seg-id decode,
`SEG_LVL_ALT_Q`, `intra_only`, and `reset_frame_context < 3` (rfc==2 seen) all have official
end-to-end coverage; `SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP` remain unit-test-only
since no official or readily-encodable IVF vector exercises them.

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
`[dev-dependencies]` entry; that was replaced because this repo's zero-dependency policy
(README: "zero dependencies, including dev-dependencies ... only the Rust standard
library") extends to dev-dependencies too. ffmpeg/ffmpeg-sys bindings were never an
option for the same reason (and would pull in a C toolchain besides).

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
