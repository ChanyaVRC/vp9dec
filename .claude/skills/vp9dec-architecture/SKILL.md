---
name: vp9dec-architecture
description: Architecture and navigation guide for the vp9dec VP9 decoder — the mental model, the decode pipeline and module map (where each piece of decode logic lives), the public API and core output types, and the load-bearing invariants and bit-exactness gotchas. Load before making or reviewing a non-trivial change to the decoder.
---

# vp9dec architecture & navigation

A from-scratch, clean-room VP9 decoder (Rust, std-only). Decode logic follows the VP9 Bitstream
& Decoding Process Specification v0.7; no other decoder's source is consulted. Its whole value
is bit-exact conformance — the `verify-vp9dec` skill is the acceptance gate.

This is a navigation aid: where things live and what not to break. For the full descriptive
module map with spec-section references, see README.md "Current architecture"; for the
still-relevant design rationale, landmines, and known gaps, `docs/implementation-notes.md`.

## Mental model

One `Decoder` is stateful for the life of a stream: VP9 carries reference frames, probability
tables, loop-filter / segmentation deltas, and frame-context slots across frames. Feed it one
container chunk at a time, in bitstream order:

```rust
let mut dec = Decoder::new();
let outputs: Vec<DecodedFrame> = dec.decode_frame(chunk)?;   // one call per IVF chunk
```

A single chunk may pack several VP9 frames (the **superframe** mechanism — e.g. a hidden altref
followed by the visible frame). `decode_frame` splits it and returns one `DecodedFrame` per
constituent frame, in order.

## Public API (the current output types)

- `DecodedFrame { info: Option<FrameDecodeInfo>, frame: Option<Frame> }`
  - `info` is `None` only on a `show_existing_frame` chunk (no header parsed on that path).
  - `frame` is `None` for a hidden frame (`show_frame == 0`).
- `Frame { width, height, bit_depth: u8, subsampling_x: u32, subsampling_y: u32, y, u, v }`,
  cropped to display size. The planes are `PlaneData`, **not** `Vec<u8>`:
  - `enum PlaneData { U8(Vec<u8>), U16(Vec<u16>) }` — 8-bit streams give `U8`; 10/12-bit
    (profile 2/3) give `U16` (samples `0..=1023` / `0..=4095`).
  - `PlaneData::as_u8()` **panics on a U16 plane** — an 8-bit-only consumer must gate on
    `bit_depth == 8` first. `subsampling_x/y` give the chroma ratio (`0,0` = 4:4:4, `1,1` = 4:2:0).
- `DecodeError`: `Header` / `CompressedHeader` / `Tile` (wrapping the parse-layer error),
  `TruncatedFrame`, `MissingReferenceFrame`. There is **no** bit-depth error — all four profiles
  decode.
- `ivf::{IvfReader, IvfHeader, IvfFrame, write_ivf}` is the only other public module. Everything
  else is `#[doc(hidden)] pub` for the pure-std integration tests, not a stable API.

## Where decode logic lives (to change X, go here)

Pipeline order, hub-first:

- **Bit / arithmetic readers** — `bit_reader` (uncompressed-header `f(n)`/`s(n)`), `bool_coder`
  (arithmetic decoder for the compressed header + all tile data, plus `read_tree`).
- **Headers** — `header::parse_uncompressed_header` (frame size/type, loop-filter / quant /
  segmentation params, cross-frame `PersistentState`); `compressed_header::parse_compressed_header`
  (`tx_mode` + probability forward updates). The 4 `frame_context_idx` slots live in
  `FrameContextStore`.
- **Tile decode hub** — `tile.rs` (`decode_partition` / `decode_block`) dispatching to
  `tile/{mode_info, ref_ctx, mv_pred, residual}`. `residual.rs` runs per-plane intra/inter
  prediction + token decode + reconstruction, and holds the per-frame dequant table.
- **Prediction / transform / quant** — `predict::{predict_intra, predict_inter}` (motion comp:
  MV selection, edge clamp, reference-frame scaling, subpel interpolation, compound); `transform`
  (inverse DCT/ADST/WHT); `quant` (dequant); `scan` (scan-order tables).
- **Loop filter** — `loop_filter` (deblocking, once per frame after all tiles).
- **DPB / superframe** — `dpb::Dpb` (8 reference slots; also serves `show_existing_frame`);
  `superframe::split_superframe`.
- **Pure tables** (no decode logic) — `common`, `subpel`, `mv_ref_tables`, `prob_tables`.
- **SIMD** — `simd.rs` (x86_64 AVX2, runtime-detected). Output **must equal** the scalar path;
  `VP9DEC_NO_SIMD=1` forces scalar and the sweep must pass identically in both configs.
- **Encoder mirrors** (test-only, `feature = "test-support"`) — `test_support` hand-builds
  synthetic bitstreams for round-trip tests; never in a normal build.

## Gotchas (bit-exactness landmines)

- **High-bit-depth scaling in `loop_filter`.** Its constants are 8-bit and scale by
  `<< (bit_depth - 8)` (identity at 8-bit). Any loop-filter change must be re-checked at
  8 / 10 / 12-bit, not just 8-bit.
- **Only the ADST SIMD kernels are 8-bit-gated.** The AVX2 inter-prediction (unscaled and
  reference-scaled), loop-filter, and DCT_DCT-transform kernels run at **all** bit depths, each
  depth-aware by a different mechanism (a `max_val` clip bound / `<< (bit_depth-8)` constant
  scaling / i64-widened butterfly products). The ADST-containing transform kernels dispatch only
  at `bit_depth == 8` (their unrounded `S` array exceeds i32 lane storage at 10/12-bit) and MUST
  stay gated. Any new kernel with hardcoded 8-bit constants or i32-tight arithmetic needs the
  same gate until made depth-aware — see the landmines in `docs/implementation-notes.md`.
- **Do not "fix" the sub-8x8 chroma MV block index in `residual.rs`.** `(y * num4x4w + x)` is
  bit-exact-correct; a plausible, spec-grounded "correction" to it once *regressed* the official
  4:2:2 vector. Verify against the sweep before touching it.
- **A chunk can produce 0, 1, or many `DecodedFrame`s** (superframes, hidden frames,
  `show_existing_frame`). Never assume one-in-one-out.

## After a change

Run the **`verify-vp9dec`** acceptance gate: default suite green, the full official sweep
315/315 in **both** SIMD configs (proving 8-bit output stayed bit-exact), the ffmpeg
cross-decode, and lint/fmt. Empirical bit-exactness beats static reasoning — when the sweep and
a hypothesis disagree, the sweep wins.
