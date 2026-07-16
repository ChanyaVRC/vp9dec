# vp9dec

A fully from-scratch VP9 video decoder (Rust, zero dependency crates).

## Purpose

With an eye toward eventual integration into the visual novel engine [Noiria](../noiria),
this implements a clean-room decoder for VP9 (a royalty-free video codec). The decoder itself --
everything under `src/` -- depends on no external crates and is implemented using only the Rust
standard library, with no runtime dependencies whatsoever. Test and verification tooling may use
`[dev-dependencies]` where genuinely useful; in practice the only one used is a self-referencing
dev-dependency on this crate itself (`vp9dec = { path = ".", features = ["test-support"] }`, see
`Cargo.toml`), which exposes test-only encoder helpers to the integration tests under `tests/` --
still not an external crate, and it has no effect on a normal `cargo build`.

The primary reference is the [VP9 Bitstream & Decoding Process Specification v0.7](
https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf)
(Google, February 22, 2017 edition). No existing OSS implementation (libvpx etc.) source code
is consulted (clean-room implementation).

## Current architecture

### Public API

- `Decoder`: a stateful decoder for one VP9 stream (VP9 carries reference frames, probability
  tables, and other state across frames -- see the doc comment on `Decoder` in `src/lib.rs` for
  the full list). `Decoder::new()` then repeated calls to
  `Decoder::decode_frame(&mut self, chunk: &[u8]) -> Result<Vec<DecodedFrame>, DecodeError>`,
  one call per container chunk (e.g. one IVF frame), in bitstream/decode order. A chunk may pack
  more than one VP9 frame via the "superframe" mechanism (common in real-world streams, e.g. a
  hidden altref frame followed by a visible frame); `decode_frame` splits it internally and
  returns one `DecodedFrame` per constituent VP9 frame, in order.
- `DecodedFrame { info: Option<FrameDecodeInfo>, frame: Option<Frame> }`: `info` is `None` only
  for a `show_existing_frame` chunk (no uncompressed header is parsed on that path); `frame` is
  `None` for a hidden frame (`show_frame == 0`).
- `Frame { width, height, y, u, v }`: one decoded picture, cropped to display size, as row-major
  YUV420 `Vec<u8>` planes.
- `FrameDecodeInfo`: read-only per-frame decode statistics (`intra_only`, `frame_is_intra`,
  `reset_frame_context`, `segmentation_enabled`, `seg_features_active`) for observation only
  (e.g. test assertions that a stream actually exercised a given decode path) -- has no effect
  on decode behavior.
- `DecodeError`: `Header`/`CompressedHeader`/`Tile` (wrapping the corresponding parse-layer
  error), `TruncatedFrame`, `UnsupportedBitDepth(u8)`, `MissingReferenceFrame`.
- `ivf`: the only other module in the public surface. `IvfReader`/`IvfHeader`/`IvfFrame`/
  `IvfError` for reading an IVF container, and `write_ivf(fourcc, width, height,
  timebase_den, timebase_num, frames) -> Vec<u8>` (the inverse), used by
  `examples/webm_to_ivf.rs` and by the test suite's synthetic-vector dump harness.

Every other top-level module (`bit_reader`, `bool_coder`, `common`, `compressed_header`,
`counts`, `dpb`, `framebuffer`, `header`, `loop_filter`, `mv_ref_tables`, `predict`,
`prob_tables`, `quant`, `scan`, `subpel`, `superframe`, `tile`, `transform`) is `#[doc(hidden)]
pub` -- public only so the pure-std integration tests under `tests/` can reach them, not a
stable API. `test_support` (encoder-side mirrors of the decoders, used to hand-build synthetic
bitstreams) is gated behind `#[cfg(any(test, feature = "test-support"))]`.

### Module map

The decode pipeline, roughly in the order data flows through it:

- **`bit_reader` / `bool_coder`**: the two low-level bit readers VP9 uses. `bit_reader::BitReader`
  reads the uncompressed header's `f(n)`/`s(n)` descriptors (spec §9.1) via plain MSB-first bit
  reads; `bool_coder::BoolDecoder` is VP9's arithmetic (bool) decoder (spec §9.2), used for the
  compressed header and all tile data, plus `read_tree` (spec §9.3.3) for tree-typed syntax
  elements.
- **`header` / `compressed_header`**: `header::parse_uncompressed_header` (spec §6.2/§7.2) reads
  frame type/size, loop filter/quantization/segmentation parameters, and everything else in the
  uncompressed header, threading `PersistentState` (reference frame sizes, loop filter deltas,
  segmentation feature state -- the cross-frame state spec §7.2 requires) in from the caller.
  `compressed_header::parse_compressed_header` (spec §6.3) reads `tx_mode` and every
  probability-table forward update, working against a `FrameContext` (`CompressedHeaderProbs`)
  supplied by the caller and returning the updated one; `FrameContextStore` holds the 4
  `frame_context_idx`-addressed slots (spec §7.1.2 `load_probs`/`save_probs`).
- **`tile` and `tile/{mode_info, ref_ctx, mv_pred, residual}`**: `tile.rs` is the hub --
  `TileDecoder`, `decode_tiles`/`decode_partition`/`decode_block` (spec §6.4) -- dispatching into
  its submodules: `mode_info` (segment id, skip, tx_size, `is_inter`/ref-frame/mode-info syntax,
  and motion vector syntax, spec §6.4.6-§6.4.20), `ref_ctx` (pure neighbor-context derivation for
  `comp_mode`/`comp_ref`/`single_ref_p1`/`single_ref_p2`, spec §9.3.2), `mv_pred` (motion vector
  prediction, spec §6.5 `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs`), and `residual`
  (`residual()` itself, spec §6.4.21: per-plane intra/inter prediction dispatch, token decoding
  spec §6.4.24-§6.4.26, and reconstruction spec §8.6.2, plus the per-frame dequantization table).
- **`predict` / `transform` / `quant` / `scan`**: `predict::predict_intra` (spec §8.5.1, all 10
  VP9 intra modes) and `predict::predict_inter` (spec §8.5.2, motion compensation: MV selection,
  edge clamping, reference-frame scaling, and 8-tap/bilinear subpel interpolation, including
  compound prediction). `transform` implements the inverse DCT/ADST (4/8/16/32-point) and Walsh-
  Hadamard transforms (spec §8.7). `quant` is dequantization (spec §8.6.1). `scan` holds the
  coefficient scan-order tables (spec §10.1).
- **`loop_filter`**: the deblocking filter (spec §8.8), applied once per frame after tile
  decoding, using the actual `ref_frame`/`y_mode` values (so it works identically for intra and
  inter frames) and the loop filter deltas persisted across frames.
- **`dpb` / `superframe`**: `dpb::Dpb` is the 8-slot reference frame buffer (spec §8.10), storing
  pixel data already cropped to display size; also serves `show_existing_frame` (spec §8.9).
  `superframe::split_superframe` implements the VP9 superframe-index format, splitting one
  container chunk into its constituent VP9 frames -- called internally by
  `Decoder::decode_frame`.
- **`common` / `subpel` / `mv_ref_tables` / `prob_tables`**: shared constant tables with no
  decode logic of their own. `common` holds the handful of constants/helpers needed by two or
  more otherwise-unrelated modules (`MAX_SEGMENTS`, `INTRA_FRAME`, `get_uv_tx_size`). `subpel`
  holds the subpel filter coefficients and motion-compensation constants (spec §8.5.2.4).
  `mv_ref_tables` holds the constant tables `find_mv_refs` uses (spec §6.5.1). `prob_tables` is
  the bulk of the default probability tables and decode trees (spec §9.3.1, §10.2-§10.5).
- **`test_support`** (feature-gated): `BoolEncoder`/`BitWriter`, the encoder-side mirrors of
  `bool_coder`/`bit_reader` used to hand-build synthetic VP9 bitstreams for round-trip tests.
  Compiled only under `cfg(test)` or the `test-support` feature -- never in a normal build.

### Current decode support

8-bit (`BitDepth == 8`) key frames, intra-only frames, and inter frames, 4:2:0 chroma
subsampling, profile 0. Full segmentation (segment-id decode including temporal prediction,
and all four `SEG_LVL_*` features), superframes, and `show_existing_frame` are all supported.
See the status table below for what's verified against an external oracle vs. unit-test-only.

## Status / limitations

### What's proven (verified against an external oracle, not just this crate's own tests)

| Area | Evidence |
| --- | --- |
| Key frame + inter frame decode, motion compensation, probability adaptation, DPB, loop filter | 5 official libvpx conformance vectors, every displayed frame bit-exact against the official `.ivf.md5`: `vp90-2-12-droppable_1` (99/99 frames), `vp90-2-09-subpixel-00` (20/20), `vp90-2-15-segkey` (1/1), `vp90-2-15-segkey_adpq` (150/150), `vp90-2-16-intra-only` (7/7) |
| Segmentation: seg-id decode, `SEG_LVL_ALT_Q` | Included in the `vp90-2-15-segkey*` bit-exact matches above |
| `intra_only` frames, `reset_frame_context` values 0 and 2, superframe splitting | Included in the `vp90-2-16-intra-only` bit-exact match above (a real 4-frame superframe) |
| Segmentation: `SEG_LVL_ALT_L`, `SEG_LVL_REF_FRAME`, `SEG_LVL_SKIP` (no official vector exists for these -- see below) | Synthetic round-trip vectors (`tests/synthetic_seg_test.rs`) cross-decoded byte-identically by two independent third-party VP9 decoders (ffmpeg's `libvpx-vp9` and its native `vp9`), 8/8 `[xdecode]` checks (4 scenarios x 2 decoders) |
| Loop filter, including the `SEG_LVL_ALT_L` level-override path | Exercised by every vector above; the `SEG_LVL_ALT_L` synthetic vectors additionally pin an exact hand-derived pixel value (not just "output changed") that matches both ffmpeg decoders |
| Motion compensation, including the reference-scaling bound derivation (`MAX_INTERMEDIATE_HEIGHT`, the largest intermediate buffer a scaled reference can require per spec §8.5.2.3) | Derived from the spec's integer arithmetic and guarded by `debug_assert!`s; exercised (unscaled case) by every vector above |

### Known limits

| Limit | Detail |
| --- | --- |
| 8-bit only | `Plane` (`src/framebuffer.rs`) is fixed to `u8`; a 10-bit/12-bit stream returns `DecodeError::UnsupportedBitDepth` rather than decoding |
| Reference-frame scaling unverified end-to-end | `scale_mv_for_plane` (spec §8.5.2.3) is implemented generically per the spec formula, and its buffer-sizing bound is derived and `debug_assert!`-checked (see above), but no available test vector has a reference frame whose size actually differs from the current frame, so the scaling *path itself* has never produced a bit-exact match against an external decoder |
| M4 (full official vector sweep) pending | Only 5 of the full libvpx `vp90-2-*` vector set are exercised so far (see `scripts/fetch-vectors.{sh,ps1}` to fetch more) |

## Tests & verification

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
```

`cargo test` runs 6 binaries: the library's own unit tests (`src/`, colocated `#[cfg(test)]
mod tests` next to the code they check), `tests/api_test.rs`, `tests/conformance_test.rs`,
`tests/synthetic_seg_test.rs`, the `decode_to_png` example's own `#[cfg(test)]` tests (PNG/CRC32/
Adler32 encoding checks), and the (currently empty) doc-tests.

- **`tests/api_test.rs`**: parse-layer probes against the two local official vectors, using
  `Decoder` and the lower-level parsing entry points directly -- IVF container + uncompressed
  header fields, `compressed_header`/`decode_tiles` read-through without panicking, and a
  plausibility check on `decode_frame`'s Y-plane output (non-degenerate variance/range).
- **`tests/conformance_test.rs`**: the bit-exact MD5 checks in the table above, plus a
  `[coverage]` line per segmentation/intra-only vector (via `FrameDecodeInfo`) confirming the
  vector actually exercises the decode path it's meant to prove, not just that the output
  happens to match; also carries the from-scratch MD5 (RFC 1321) implementation's own unit
  tests (`mod md5_tests`).
- **`tests/synthetic_seg_test.rs`**: the `SEG_LVL_ALT_L`/`SEG_LVL_REF_FRAME`/`SEG_LVL_SKIP`
  synthetic round-trip vectors and their ffmpeg cross-decode check (see the status table above
  and `docs/implementation-notes.md` for what this test does and doesn't prove). Also has an
  env-gated dump test (`dump_synthetic_ivf_for_external_cross_decode`) for producing the raw
  `.ivf`/`.yuv` files that cross-decode check consumes.
- **`tests/common/`**: shared test infrastructure, not a test binary itself -- `mod.rs` (vector/
  `.ivf.md5` loading with the skip-if-absent convention below, I420 byte layout), `encoder.rs`
  (the ~400-line synthetic VP9 bitstream encoder `tests/synthetic_seg_test.rs` drives), `md5.rs`
  (the MD5 implementation).

### Getting the test vectors

Test vectors and `.ivf.md5` files aren't included in the repository (excluded via
`.gitignore`), so they must be downloaded before the corresponding tests can run for real. If
they're missing, the affected test skips cleanly via early return + `eprintln!` rather than
failing, so `cargo test` stays green either way -- but you'll want them present to actually
exercise the conformance checks.

```sh
bash scripts/fetch-vectors.sh
```

```powershell
pwsh scripts/fetch-vectors.ps1
```

Both scripts are idempotent (skip any file already present) and manifest-driven
(`scripts/vectors.txt`): they download `<name>.ivf`/`<name>.ivf.md5` directly for vectors that
ship as IVF, and for the three vectors libvpx only ships as `.webm`, they download the `.webm` +
`.webm.md5` and remux to `.ivf` via `cargo run --example webm_to_ivf` (container change only, no
re-encode), then copy the `.webm.md5` alongside it as `.ivf.md5` (the MD5s are of the decoded
pixel output, not the container, so they carry over unchanged).

### External cross-decode (ffmpeg)

ffmpeg is not a project dependency -- nothing in `src/`/`examples/` links against it -- but its
`libvpx-vp9` and native `vp9` decoders serve as independent oracles for the synthetic vectors
(see the status table above) and as a debugging aid for byte-level diffs against a known-correct
decode. `synthetic_streams_cross_decode_against_ffmpeg` (`tests/synthetic_seg_test.rs`) shells
out to the ffmpeg *binary* via `std::process::Command`, located via the `VP9DEC_FFMPEG`
environment variable or `ffmpeg` on `PATH`; if no ffmpeg binary is found, it prints a `[skip]`
line and passes trivially.

```sh
VP9DEC_FFMPEG=/path/to/ffmpeg cargo test --test synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture
```

```powershell
$env:VP9DEC_FFMPEG = "C:\path\to\ffmpeg.exe"; cargo test --test synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture
```

To dump the same synthetic vectors' raw `.ivf`/decoded `.yuv` for manual inspection with any
other external decoder, set `VP9DEC_DUMP_DIR` (a no-op unless set, so a plain `cargo test` run
stays green):

```sh
VP9DEC_DUMP_DIR=/some/dir cargo test --test synthetic_seg_test dump_synthetic_ivf_for_external_cross_decode -- --nocapture
```

```powershell
$env:VP9DEC_DUMP_DIR = "C:\some\dir"; cargo test --test synthetic_seg_test dump_synthetic_ivf_for_external_cross_decode -- --nocapture
```

### PNG dump (for visual inspection)

`examples/decode_to_png.rs` decodes an `.ivf`, converts YUV to RGB using BT.601 (limited
range), and writes the result to `target/dump/` as a PNG (its own from-scratch, dependency-free
encoder -- see the file's module doc). With no arguments it dumps the first (key) frame of the
two local non-webm vectors; given a vector name and an IVF frame number, it dumps the first
displayed frame at or after that number (for inspecting an inter frame with motion
compensation):

```sh
cargo run --example decode_to_png
cargo run --example decode_to_png -- vp90-2-12-droppable_1 50
```

## History

For the milestone-by-milestone build narrative (M1 through M3 second half) this section used
to contain, see [docs/history.md](docs/history.md). For dated design decisions, tradeoffs, and
fixes made along the way (including the six design-debt waves that produced the architecture
described above), see [docs/implementation-notes.md](docs/implementation-notes.md).

## License

MIT
