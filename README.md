# vp9dec

A fully from-scratch VP9 video decoder (Rust, zero dependency crates).

## Purpose

With an eye toward eventual integration into the visual novel engine [Noiria](../noiria),
this implements a clean-room decoder for VP9 (a royalty-free video codec). It depends on no
external crates (zero dependencies, including dev-dependencies) and is implemented using only
the Rust standard library.

The primary reference is the [VP9 Bitstream & Decoding Process Specification v0.7](
https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf)
(Google, February 22, 2017 edition). No existing OSS implementation (libvpx etc.) source code
is consulted (clean-room implementation).

## Phased plan

| Milestone | Content | Status |
| --- | --- | --- |
| M1 | Container (IVF) parser, bool decoder, uncompressed frame header parsing | Done |
| M2 | Key frame decoding via intra prediction (compressed header, tiles, token decoding, transform, quantization, reconstruction) | Done (except the loop filter) |
| M2b | Loop filter (deblocking filter, spec section 8.8) + official conformance verification | Done |
| M3 first half | Inter-frame bitstream decoding (header, probability tables, mode info, MV, residual tokens; up to but not including motion compensation) | Done |
| M3 second half | Motion compensation, subpixel interpolation, reference frame management, probability adaptation (forward/backward), full-frame MD5 conformance | **Done** |
| M4 | Full pass of VP9 conformance test vectors (expand beyond the 2 local vectors to the official vector set) | Not started |

`decode_keyframe()` (in `src/lib.rs`) decodes a key frame all the way through, producing a
YUV420 `Frame` with the loop filter applied and cropped to display size. To decode multiple
frames in sequence (including inter frames), use `Decoder::decode_frame()` (see the "M3
second half" section below). For our two local test vectors, we've confirmed that the decoded
result of **every displayed frame** (the Y->U->V concatenated I420 byte string) has an MD5
that exactly matches the official `.ivf.md5` distributed by libvpx (`tests/conformance_test.rs`,
details below. `vp90-2-12-droppable_1`: 99/99 frames matched, `vp90-2-09-subpixel-00`: 20/20
frames matched).

## What was implemented in M1

- `src/ivf.rs`: IVF container parser (reads the 32-byte header + 12-byte frame headers).
- `src/bool_coder.rs`: VP9's arithmetic coder (bool coder) decoder (spec section 9.2).
  A matching encoder was implemented in the tests for verification, confirming round-trip
  correctness.
- `src/header.rs`: Parsing of the uncompressed frame header (uncompressed_header, spec
  section 6.2). At the M1 stage this only supported key frames (inter frames / intra-only
  frames were added in M3 first half; see the "M3 first half" section below for details).

## M2 (intra decoding of key frames)

The following is implemented (all of it targets key frames / `FrameIsIntra == 1` only; syntax
related to inter prediction is never read per spec, so it's unimplemented).

- `src/bool_coder.rs`: `read_tree` (spec section 9.3.3, "Tree decoding process").
- `src/prob_tables.rs`: Tree definitions (`PARTITION_TREE`/`TOKEN_TREE`/`INTRA_MODE_TREE`
  etc.) and the full set of default probability tables. `KF_PARTITION_PROBS`/`KF_Y_MODE_PROBS`/
  `KF_UV_MODE_PROBS` come from spec section 10.4; `DEFAULT_COEF_PROBS`/`DEFAULT_TX_PROBS`/
  `DEFAULT_SKIP_PROB`/`INV_MAP_TABLE` from spec sections 10.5 and 6.3.5; the block size
  conversion tables and `SS_SIZE_LOOKUP` from spec sections 10.2 and 6.4.23; `PARETO_TABLE`
  (128x8) from spec section 10.3; `COEFBAND_4X4`/`coefband_8x8plus()`/`ENERGY_CLASS`/
  `EXTRA_BITS`/`CAT_PROBS`/`mode2txfm_map()` transcribed from spec sections 6.4.24-6.4.26.
  For tables with large numeric content (`DEFAULT_COEF_PROBS`, `PARETO_TABLE`), to avoid
  manual transcription errors, the numbers were mechanically extracted with a `grep -oE`
  regex from spec PDF text produced by `pdftotext -layout`, then converted directly into Rust
  array literals (`coefband_8x8plus` was extracted the same way from the real 1024-element
  data; after confirming that the trailing 1003 elements are all `5`, it was implemented
  compactly as a function rather than an array).
- `src/compressed_header.rs`: `compressed_header()` (spec section 6.3).
- `src/tile.rs`: In addition to `decode_tiles`/`decode_tile`/`decode_partition`/`decode_block`/
  `intra_frame_mode_info` (spec section 6.4), this task **fully implemented `residual()`
  (spec section 6.4.21)**. For each plane, it:
  1. Determines chroma transform size and block size via `get_uv_tx_size()`/
     `get_plane_block_size()` (spec sections 6.4.22-6.4.23).
  2. Performs intra prediction via `predict_intra()` (`src/predict.rs`, spec section 8.5.1).
     This always runs regardless of `skip` (per spec, prediction happens outside the `!skip`
     check).
  3. Only when `skip == 0`: performs token decoding via `tokens_and_reconstruct()`
     (`tokens()`, spec section 6.4.24) -> determines `get_scan()`/`TxType` (spec section
     6.4.25) -> inverse quantization, inverse transform, and reconstruction
     (`reconstruct()`, spec section 8.6.2, using `src/quant.rs`/`src/transform.rs`).
  4. Updates `AboveNonzeroContext`/`LeftNonzeroContext` (the loop at the end of spec section
     6.4.21; always updated with the `nonzero` value, even for cases not read due to `skip`
     or being off the edge of the frame).
- `src/framebuffer.rs` (new): `Plane` (equivalent to `CurrFrame[plane]`). Frame buffers are
  allocated rounded up to superblock boundaries (`Sb64Cols*64`/`Sb64Rows*64`, chroma after
  subsampling). Reason: `predict_intra`/`reconstruct` writes can slightly exceed
  `(MiCols*8, MiRows*8)` (reads are always clipped via `Min(maxX, ...)`, but writes such as
  assignments to `pred[i][j]`/`Dequant[i][j]` are not clipped).
- `src/predict.rs` (new): `predict_intra()` (spec section 8.5.1). Implements all 10 of VP9's
  intra modes (`DC`/`V`/`H`/`D45`/`D135`/`D117`/`D153`/`D207`/`D63`/`TM`; the smooth-family
  filters don't exist in VP9, so they aren't implemented). Handles `aboveRow`/`leftCol`
  availability checks, frame-edge clamping, and `notOnRight` (allowing right-side references
  only for 4x4 transforms) exactly per spec.
- `src/lib.rs`: The public API `decode_keyframe(frame_data: &[u8]) -> Result<Frame, DecodeError>`.
  Runs uncompressed header -> compressed header -> tile decoding end-to-end and returns a
  `Frame { width, height, y, u, v }` (cropped to display size, per the output process in spec
  section 8.9).

### Known limitations

- Only 8-bit (`BitDepth == 8`) is supported. Since `Plane` is fixed at `u8`, 10-bit/12-bit
  frames cause `decode_keyframe` to return `DecodeError::UnsupportedBitDepth`.
- Segmentation is fully supported (segment-id decoding including temporal prediction, and
  the SEG_LVL_ALT_Q / SEG_LVL_ALT_L / SEG_LVL_REF_FRAME / SEG_LVL_SKIP features). It is
  unit-tested only: no official IVF-form segmentation conformance vector exists upstream
  (the `vp90-2-15-segkey*` vectors are `.webm`-only); see `docs/implementation-notes.md`.
- The judgment call regarding the known erratum in how spec section 9.3.2 describes the
  probability selection process for `partition` is unchanged from M1. See the comment on
  `read_partition` in `src/tile.rs` for details.

## M2b (loop filter + official conformance verification)

The following is implemented.

- `src/loop_filter.rs` (new): the deblocking filter (spec section 8.8). Implements, in
  straightforward integer arithmetic exactly matching the spec's pseudocode: the overall
  frame traversal order (superblocks in raster order -> Y/U/V -> vertical edges ->
  horizontal edges, per the pseudocode at the start of spec section 8.8), filter strength
  calculation (`build_lvl_lookup`, spec section 8.8.1, "Loop filter frame init process"),
  edge determination (excluding block boundaries, transform block boundaries, and frame
  edges, spec section 8.8.2), filter size determination (spec section 8.8.3), adaptive
  filter strength (`limit`/`blimit`/`thresh`, spec section 8.8.4), and the filter itself
  (narrow filter = 4-tap, wide filter = 8/16-tap, including flat/flat2 determination, spec
  section 8.8.5). Since only key frames are targeted, `isIntra` is hardcoded to true and
  `modeType` to 0 (`MiInfo` doesn't carry `ref_frame` yet; see the M3 handoff note).
  Invoked from `src/tile.rs::TileDecoder::apply_loop_filter()`, and applied by
  `src/lib.rs::decode_keyframe` right after tile decoding and before cropping.
- `src/md5.rs` (new): a from-scratch MD5 (RFC 1321) implementation (per the zero-dependency
  policy). Unit-tested against known vectors (empty string, `"abc"`, `"message digest"`, etc.,
  the values listed in RFC 1321).
- `tests/conformance_test.rs` (new): compares the official `.ivf.md5` distributed by libvpx
  (downloaded into `tests/vectors/`; download steps below) against the MD5 of
  `decode_keyframe`'s output for the first key frame (the Y->U->V concatenated I420 byte
  string). **Both test vectors (`vp90-2-12-droppable_1` and `vp90-2-09-subpixel-00`) have
  been confirmed to match exactly**. In particular, `vp90-2-09-subpixel-00`'s key frame looks
  like pseudo-random noise (see `target/dump/*.png`), which left some doubt at the M2 stage
  about whether the decode result was correct; matching the official MD5 confirmed that the
  decode is in fact correct (the effect of the subpixel interpolation filter is presumably
  only visible in later inter frames — worth a fresh visual inspection on an intermediate
  frame with motion compensation once M3 is implemented).

## M3 first half (inter-frame bitstream decoding)

This implements everything needed to correctly read an inter frame's bitstream from start to
finish, except for motion compensation / subpixel interpolation itself (pixel generation).
The following is implemented.

- `src/header.rs`: Fully implemented the non-key-frame branch of `uncompressed_header()`
  (spec section 6.2). Added `intra_only`/`reset_frame_context`/`refresh_frame_flags` (for
  inter frames)/`ref_frame_idx`/`ref_frame_sign_bias`/`frame_size_with_refs()` (spec section
  6.2.5)/`allow_high_precision_mv`/`read_interpolation_filter()` (spec section 6.2.7).
  Since the `RefFrameWidth`/`RefFrameHeight` that `frame_size_with_refs()` reads are
  cross-frame state, `parse_uncompressed_header()` is designed to receive them from the
  caller as `&[(u32,u32); NUM_REF_FRAMES]` (the state itself is held by `Decoder`, in
  `src/lib.rs`). Since inter frames don't resend `color_config`, `Decoder` carries forward
  the value from the most recent key frame / intra-only frame.
- `src/prob_tables.rs`: Added inter-related trees (`INTER_MODE_TREE`/`INTERP_FILTER_TREE`/
  `MV_JOINT_TREE`/`MV_CLASS_TREE`/`MV_FR_TREE`) and the full set of default probability tables
  mechanically extracted from spec section 10.5 (`DEFAULT_PARTITION_PROBS`/
  `DEFAULT_Y_MODE_PROBS`/`DEFAULT_UV_MODE_PROBS`/`DEFAULT_IS_INTER_PROB`/
  `DEFAULT_COMP_MODE_PROB`/`DEFAULT_COMP_REF_PROB`/`DEFAULT_SINGLE_REF_PROB`/
  `DEFAULT_INTER_MODE_PROBS`/`DEFAULT_INTERP_FILTER_PROBS`/8 MV-related tables), plus the
  constant tables from spec section 6.5, "Motion vector prediction"
  (`MV_REF_BLOCKS`/`MODE_2_COUNTER`/`COUNTER_TO_CONTEXT`/`IDX_N_COLUMN_TO_SUBBLOCK`/
  `SIZE_GROUP_LOOKUP`). `mode2txfm_map()` was extended to also accept inter mode values
  (`NEARESTMV`..`NEWMV`) (the table in spec section 10.2 defines the full
  MB_MODE_COUNT=14 range, and all inter modes map to `DCT_DCT`).
- `src/compressed_header.rs`: Implemented `read_inter_mode_probs`/`read_interp_filter_probs`/
  `read_is_inter_probs`/`frame_reference_mode` (including `setup_compound_reference_mode`)/
  `frame_reference_mode_probs`/`read_y_mode_probs`/`read_partition_probs` (all the
  non-key-frame variants)/`mv_probs` (including `update_mv_prob`, spec sections 6.3.9-6.3.18).
  Extended `CompressedHeaderProbs` to cover "every probability table that `load_probs`/
  `save_probs` operate on" (excluding `uv_mode_probs`, since there's no update syntax for it
  and it's always the default value), and added `FrameContext` (an alias for
  `CompressedHeaderProbs`) and a 4-slot `FrameContextStore`. `parse_compressed_header_ex()`
  is the new inter-capable entry point; `parse_compressed_header()` (key-frame-only, the
  existing API) remains as a thin wrapper around it.
- `src/tile.rs`: Fully implemented `inter_frame_mode_info()` (spec section 6.4.11) and
  everything below it: `read_is_inter`/`intra_block_mode_info` (intra blocks inside an inter
  frame)/`read_ref_frames` (context derivation for `comp_mode`/`comp_ref`/`single_ref_p1`/
  `single_ref_p2`, including spec section 9.3.2)/`inter_block_mode_info`/`assign_mv`/
  `read_mv`/`read_mv_component` (spec sections 6.4.16-6.4.20). Motion vector prediction (spec
  section 6.5) is implemented as `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs`/
  `is_inside`/`get_block_mv`/`if_same_ref_frame_add_mv`/`if_diff_ref_frame_add_mv`, with the
  pure helper computations (clamping, sign inversion, threshold checks) factored out into the
  new `src/mv.rs`. `UsePrevFrameMvs` (spec section 7.2.6) is also supported, receiving the
  previous frame's `MiGrid` (equivalent to `Mvs`/`RefFrames`) via `TileDecoder::new_with_prev()`.
  `residual()` was updated to handle the `is_inter` branch (where `predict_inter` gets
  called, `TxType` determination, `coef_probs`'s `is_inter` index) and `EobTotal` (spec
  section 6.4.4; retroactively forces `skip` to 1 when
  `is_inter && subsize >= BLOCK_8X8 && EobTotal == 0`). `read_partition` keeps its existing
  interpretation of the known erratum in spec section 9.3.2, and switches between
  `KF_PARTITION_PROBS`/`partition_probs` depending on `FrameIsIntra`.
- `src/predict.rs`: Added `predict_inter_stub()` (a placeholder for spec section 8.5.2).
  Motion compensation / subpixel interpolation are not implemented; calling it does nothing.
  Per the NOTE in spec section 7.4.15, `predict_inter` has no effect on syntax decoding, so
  this doesn't affect the ability to fully read the bitstream.
- `src/loop_filter.rs`: Now that `MiInfo` carries `ref_frame`/`y_mode` (including inter
  values), removed the hardcoded `is_intra`/`modeType` and updated it to reference the actual
  values of `RefFrames[..][0]`/`YModes` per spec section 8.8.4. **The existing key-frame MD5
  conformance tests were confirmed to still pass in full** (since `isIntra`/`modeType`
  still evaluate to `true`/`0` for key frames as before, output is unchanged).
- `src/lib.rs`: Introduced a new `Decoder` that holds cross-frame state (reference frame slot
  sizes, 4 frame context slots, the previous frame's `MiGrid` for `UsePrevFrameMvs`, and the
  most recent `color_config`), enabling `Decoder::decode_frame()` to decode frames one at a
  time in sequence. The existing `decode_keyframe()` (which validates that the first argument
  is a key frame, then internally calls `decode_frame()` via a disposable `Decoder`) is kept
  as-is for backward compatibility.

The known limitations at the M3 first-half stage (motion compensation unimplemented,
probability adaptation unimplemented, loop filter deltas not carried forward, etc.) have all
been resolved in the following section, "M3 second half".

## M3 second half (motion compensation, probability adaptation, reference frame management, full-frame MD5 conformance)

All the M3 first-half carryover items have been implemented, achieving **an exact match on
the official MD5 for every displayed frame on both test vectors**
(`vp90-2-12-droppable_1`: 99/99, `vp90-2-09-subpixel-00`: 20/20). The following is implemented.

- `src/predict.rs`: `predict_inter()` (spec section 8.5.2), replacing `predict_inter_stub()`.
  Internally split into the 4 sub-steps described in the spec.
  - `select_mv()` (spec section 8.5.2.1): per-4x4-subblock MV selection. For chroma with
    `MiSize < BLOCK_8X8`, performs subsampling averaging via `round_mv_comp_q2`/
    `round_mv_comp_q4`.
  - `clamp_mv_for_plane()` (spec section 8.5.2.2): clamping at frame edges (1/16-pel
    precision, boundary computed using `INTERP_EXTEND`/`SUBPEL_BITS`).
  - `scale_mv_for_plane()` (spec section 8.5.2.3): scaling for when the reference frame and
    current frame sizes differ (`xScale`/`yScale`, `REF_SCALE_SHIFT`). Reference frame size
    always matched the current frame in our local test vectors, but since the spec's formula
    is implemented as-is, it should also work in theory when sizes differ (unverified).
  - `block_inter_predict()` (spec section 8.5.2.4): two-pass (horizontal then vertical) 8-tap
    (or `BILINEAR`) subpel interpolation filter convolution. Reads at reference frame edges
    are **clamped reads** via `Clip3(0, lastX/lastY, ...)` (not edge extension).
  - Also supports compound prediction (averaging two references, `Round2(pred0+pred1, 1)`).
  - The filter coefficient table `SUBPEL_FILTERS` (`src/prob_tables.rs`) is transcribed
    directly from spec section 8.5.2.4's `subpel_filters[4][16][8]`, extracted via
    `pdftotext -raw` (indices 0..3 correspond to `EIGHTTAP`/`EIGHTTAP_SMOOTH`/
    `EIGHTTAP_SHARP`/`BILINEAR`).
- `src/dpb.rs` (new): `Dpb`/`RefFrameData`. An 8-slot reference frame buffer (spec section
  8.10, "Reference frame update process"). `RefFrameData` holds pixel data cropped to display
  size (`RefFrameWidth`/`Height`, chroma after subsampling). This means `predict_inter`'s
  clamping logic can use `lastX`/`lastY` directly as `Plane::width/height - 1`.
  `Decoder::decode_frame()` calls `Dpb::update(refresh_frame_flags, ...)` after tile decoding
  and loop filter application. `show_existing_frame` pulls the relevant slot from the DPB and
  assembles it into a `Frame` (`DecodeError::MissingReferenceFrame` only occurs when that
  slot is empty, which doesn't happen for conformant bitstreams). Whether scaling is needed
  when a reference frame's size differs from the current frame couldn't be exercised with our
  local test vectors, since they were always the same size (`scale_mv_for_plane` itself is
  implemented generically per the spec formula).
- `src/counts.rs` (new): probability adaptation (spec section 8.4).
  - `Counts`: every counter array listed in spec section 8.3, "Clear counts process"
    (`counts_partition`/`counts_intra_mode`/`counts_uv_mode`/`counts_skip`/
    `counts_is_inter`/`counts_comp_mode`/`counts_comp_ref`/`counts_single_ref`/
    `counts_inter_mode`/`counts_interp_filter`/`counts_tx_size`/`counts_mv_*`/
    `counts_token`/`counts_more_coefs`). Added increment logic at each syntax element read
    site in `src/tile.rs` (`read_partition`/`read_skip`/`read_tx_size`/`read_is_inter`/
    `intra_block_mode_info`/`read_ref_frames`/`inter_block_mode_info`/`read_mv`/
    `read_mv_component`/`tokens_and_reconstruct`). Following the general rule in spec
    section 9.3 that "for tree-typed syntax elements, counting always happens even when the
    value is determined without reading any bits", `partition` is counted on every branch,
    including the case where `hasRows`/`hasCols` are both false and the value is
    unconditionally `PARTITION_SPLIT`.
  - `merge_prob`/`merge_probs` (spec sections 8.4.1-8.4.2), `adapt_coef_probs` (spec section
    8.4.3), `adapt_noncoef_probs` (spec section 8.4.4).
  - **A note on the "special case" for `more_coefs` (end of spec section 9.3.4)**: the spec
    PDF states that "`more_coefs` has special handling described at the end of this section",
    but the referenced text is actually missing across all three of `pdftotext`'s `-raw`/
    `-layout`/default modes (presumably an omission in the v0.7 draft). This implementation
    adopts the only interpretation that logically follows consistently from the general rule
    at the start of section 9.3 (tree-typed elements are counted even when not read): in
    `tokens()` (spec section 6.4.24), even when `checkEob == 0` (i.e. `more_coefs` isn't
    actually read because the previous token was `ZERO_TOKEN`), `counts_more_coefs[...][1]`
    is still incremented as if the value were 1 ("more follows"). This interpretation is
    empirically supported by the **exact full-frame MD5 match** on both test vectors (see
    the comment in `src/tile.rs::tokens_and_reconstruct`).
  - The `refresh_probs()`-equivalent logic (spec sections 6.1.2, 7.1.2) in
    `Decoder::decode_frame()`: note that `load_probs(ctx)` refers to restoring every table
    **except** `tx_probs`/`skip_prob` to their pre-forward-update values (`starting_probs`)
    (whereas `compressed_header()`'s forward update includes `tx_probs`/`skip_prob` too, but
    `refresh_probs()`'s `load_probs` leaves those two at their post-forward-update values).
    `adapt_coef_probs()` is applied to `coef_probs` in that state, and only when
    `FrameIsIntra == 0` is it followed by `load_probs2(ctx)` (which restores
    `tx_probs`/`skip_prob` to their pre-forward-update values too) -> `adapt_noncoef_probs()`.
    Misunderstanding this two-stage structure — "restore to pre-forward-update values, then
    backward-adapt" — and instead adapting against the post-forward-update values won't
    surface as a problem for a single key frame (since backward adaptation never fires), but
    will reliably fail multi-frame MD5 conformance. This implementation holds onto
    `starting_probs` (the pre-forward-update values) inside `Decoder::decode_frame()` and
    follows the above procedure exactly.
  - `uv_mode_probs` has no forward update syntax in `compressed_header()` (never updated by
    the bitstream), but spec section 8.4.4's `adapt_noncoef_probs()` does include
    `uv_mode_probs` as a backward adaptation target. In M3 first half,
    `CompressedHeaderProbs` had no such field, and reading `uv_mode` for intra blocks inside
    inter frames always used the fixed `DEFAULT_UV_MODE_PROBS` value (a bug). Added the
    `CompressedHeaderProbs::uv_mode_probs` field and fixed it to be included in
    `load_probs`/`save_probs`/backward adaptation.
- `src/header.rs`: Changed `parse_uncompressed_header()`/`parse_loop_filter_params()` to
  accept the previous frame's loop filter `ref_deltas`/`mode_deltas` (spec section 7.2, state
  that only resets on `setup_past_independence()` and persists across frames) as arguments
  (the same design pattern as `ref_frame_sizes`). `Decoder` holds this state and updates it
  every frame.
- `src/lib.rs`: Added the DPB (`dpb: Dpb`), loop filter deltas (`loop_filter_deltas`), and
  `LastFrameType` (used in `adapt_coef_probs`'s `updateFactor` computation, spec section
  8.4.3) to `Decoder`. **Changed the public API**:
  `Decoder::decode_frame(&mut self, data: &[u8]) -> Result<Option<Frame>, DecodeError>`
  (the `DecodeOutcome` enum was removed). `None` represents a "hidden frame"
  (`show_frame == 0`, i.e. a droppable/altref frame). `show_existing_frame` frames now pull
  actual pixel data from the DPB and return `Some(Frame)` (in M3 first half, this only
  returned `DecodeOutcome::ShowExisting { frame_to_show_map_idx }` without any real data).
  `decode_keyframe()` (which validates that the first argument is a key frame, then
  internally calls it via `Decoder`) remains as the backward-compatible single-shot API.
- `reset_frame_context == 2` (resetting only the relevant frame context) was a known
  limitation carried over from M3 first half (all 4 slots were always reset when
  `FrameIsIntra || error_resilient_mode`); fixed later per the spec's `setup_past_independence()`
  (see `docs/implementation-notes.md`).
- Segmentation, unsupported since M1, has since been implemented in full (see "Known
  limitations" above).

### Debugging trail (for reference)

Both vectors matched exactly on the first full-frame MD5 test run, so extensive debugging
wasn't needed. That said, the debugging strategy planned before implementation (for use in
case of a mismatch) is recorded below, since it should still be useful when adding new test
vectors in M4.
1. Compare only the 2nd frame (the first inter frame) -> isolates whether the bug is
   "self-contained within a single frame" (interpolation filter, DPB, MV prediction, etc.).
2. Disable probability adaptation (comment out the `adapt_coef_probs`/`adapt_noncoef_probs`
   calls, reverting to M3-first-half-equivalent behavior of saving the post-forward-update
   values as-is) and see how many frames match -> if more frames match, the bug is in the
   adaptation logic; if unchanged, the bug is in motion compensation / the DPB.
3. Compare predicted pixels only (forcing the residual to zero) -> isolates the correctness
   of `predict_inter` itself from that of token decoding / inverse quantization / inverse
   transform.

## Tests

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
```

### Verification with real data

`tests/header_test.rs` uses official WebM test vectors (libvpx conformance test data) to
verify the IVF parser and header parser. `tests/compressed_header_test.rs` uses the same test
vectors to verify that `compressed_header` reads through to completion and that
`decode_tiles` doesn't panic, for the first key frame. `tests/decode_test.rs` fully decodes
the first key frame end-to-end via `decode_keyframe()` (the public API) and verifies the
output is a plausible result for a real-world video vector using Y-plane statistics
(non-zero variance, not all pixels identical, `min < 50 && max > 200`).
`tests/conformance_test.rs` (added in M2b, extended in M3 second half) performs two kinds of
verification.
- `*_first_keyframe_matches_official_md5` (M2b): verifies that the MD5 of
  `decode_keyframe()`'s output for the first key frame (the Y->U->V concatenated I420 byte
  string) exactly matches the first line of the official `.ivf.md5` distributed by libvpx.
- `*_all_frames_match_official_md5` (added in M3 second half): decodes **every IVF frame** in
  order via `Decoder::decode_frame()`, and verifies that every displayed frame (where
  `Some(Frame)` is returned) has an MD5 that exactly matches the corresponding line of
  `.ivf.md5` (this won't pass unless motion compensation, probability adaptation, the DPB,
  and the loop filter's cross-frame state are all correct).
  Confirmed exact matches for `vp90-2-12-droppable_1` (99/99 output frames) and
  `vp90-2-09-subpixel-00` (20/20 output frames).

`tests/inter_frame_test.rs` (added in M3 first half) uses `Decoder::decode_frame()` to decode
**every IVF frame** of each test vector in order (including key frames, inter frames, and the
droppable frames in `vp90-2-12-droppable_1`), and verifies that `uncompressed_header` +
`compressed_header` + mode info/MV/residual tokens for every tile can be read through to
completion without panicking (confirmed for all 20 frames of `vp90-2-09-subpixel-00` and all
99 frames of `vp90-2-12-droppable_1`). Pixel correctness is verified separately in
`conformance_test.rs`; this test only checks that the bitstream can be fully read. Test
vectors and MD5 files aren't included in the repository (excluded via `.gitignore`), so they
must be downloaded beforehand using the steps below. If they haven't been downloaded, the
corresponding tests are skipped via early return + `eprintln!`, and the test suite as a whole
doesn't fail.

```sh
mkdir -p tests/vectors
curl -o tests/vectors/vp90-2-12-droppable_1.ivf \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-12-droppable_1.ivf
curl -o tests/vectors/vp90-2-09-subpixel-00.ivf \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-09-subpixel-00.ivf
curl -o tests/vectors/vp90-2-12-droppable_1.ivf.md5 \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-12-droppable_1.ivf.md5
curl -o tests/vectors/vp90-2-09-subpixel-00.ivf.md5 \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-09-subpixel-00.ivf.md5
```

(On PowerShell, use `Invoke-WebRequest -Uri <URL> -OutFile <path>` instead.)

`.ivf.md5` is in `md5sum`-compatible format (`<32-char hex>␠␠<filename>`), recording one MD5
per line for each output (displayed) frame (the Y->U->V concatenated I420 byte string).
Hidden frames with `show_frame == 0` have no line. `*_first_keyframe_matches_official_md5`
uses only the first line; `*_all_frames_match_official_md5` uses every line.

The list of test vectors is documented in the libvpx repository's
[`test/test-data.sha1`](https://github.com/webmproject/libvpx/blob/main/test/test-data.sha1).
Files matching `vp90-2-*.ivf` (without an `invalid-` prefix) are raw IVF containers; the rest
are provided in WebM container (.webm) format.

### PNG dump (for visual inspection)

`examples/decode_to_png.rs` decodes an `.ivf`, converts YUV to RGB using BT.601 (limited
range), and writes the result to `target/dump/` as a PNG. The PNG encoder is also
implemented from scratch without dependency crates (zlib only uses uncompressed "stored"
blocks; CRC-32/Adler-32 are also implemented from scratch).

With no arguments, it writes the first frame (the key frame) of both vectors to
`target/dump/<vector name>.png` (existing behavior since M2b).

```sh
cargo run --example decode_to_png
```

Given a vector name (without extension) and an IVF frame number (0-indexed, in decode order)
as arguments, it writes the first frame displayed at or after that frame to
`target/dump/<vector name>_frame<N>.png` (added in M3 second half, for visually inspecting an
intermediate frame with motion compensation. Since `Decoder` requires cross-frame state, it
decodes internally from frame 0 up through the specified frame in order).

```sh
# Example: dump around frame 50 of vp90-2-12-droppable_1
cargo run --example decode_to_png -- vp90-2-12-droppable_1 50
```

Visually inspected around frame 50 of `vp90-2-12-droppable_1` (real-world construction site
footage). `vp90-2-09-subpixel-00` is a synthetic test vector that looks like pseudo-random
noise on every frame by design, so it still looks like noise after motion compensation as
well (correctness is separately confirmed via the MD5 match; see also the existing note in
the "M2b" section).

## License

MIT
