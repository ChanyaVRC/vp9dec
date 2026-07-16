//! Synthetic, pure-std, round-trip coverage for the three segmentation features that no
//! official (or readily-encodable) libvpx IVF vector exercises: `SEG_LVL_ALT_L`,
//! `SEG_LVL_REF_FRAME`, `SEG_LVL_SKIP` (see `docs/implementation-notes.md`, "Conformance
//! coverage" section, and its 2026-07-14 follow-up entry for the full writeup).
//!
//! This file is its own tiny VP9 *encoder* (bool coder + uncompressed-header bit writer,
//! imported from `vp9dec::test_support` via the crate's `test-support` feature -- see
//! `Cargo.toml`'s self-referencing `[dev-dependencies]` entry). It builds just enough
//! bitstream -- a handful of skip=1 intra blocks and one all-forced-fields inter frame --
//! to drive the three features through `vp9dec::Decoder`, then checks the decoded output.
//!
//! This is a **self-consistent round trip** (this file's encoder <-> the crate's decoder),
//! not conformance against an official MD5: correctness of the *values themselves* (e.g.
//! that VP9's V_PRED really means "copy the row above") is exactly what's being exercised,
//! not independently rechecked against a reference encoder. What it does prove: that the
//! decoder takes the `SEG_LVL_*`-forced code paths (`seg_feature_active` in `src/tile.rs`)
//! instead of silently falling back to reading ordinary per-block bits -- if it fell back,
//! either decoding would fail outright (this bitstream supplies none of those bits) or the
//! output would diverge from the hand-derived expected pixels asserted below.
//!
//! No residual-token encoder was written: every block uses `skip = 1`, so the only pixel
//! values ever produced are straight out of intra prediction (`src/predict.rs`) or
//! zero-MV motion compensation (an exact copy) -- both are easy to hand-verify without
//! needing to also reimplement coefficient/token encoding.

use vp9dec::header::{MAX_SEGMENTS, SEG_LVL_ALT_L, SEG_LVL_MAX, SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use vp9dec::prob_tables::{
    DC_PRED, DEFAULT_PARTITION_PROBS, DEFAULT_SKIP_PROB, GOLDEN_FRAME, H_PRED, INTRA_MODE_TREE,
    KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, LAST_FRAME, PARTITION_NONE,
    PARTITION_SPLIT, PARTITION_TREE, SEGMENT_TREE, V_PRED,
};
use vp9dec::test_support::{BitWriter, BoolEncoder};
use vp9dec::{DecodedFrame, Decoder};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;

/// Finds the `(node, bit)` sequence that `BoolDecoder::read_tree` (`src/bool_coder.rs`)
/// would read, in order, to arrive at `leaf`, by walking the tree structure the same way
/// `read_tree` does. Used instead of hand-transcribing bit paths for `SEGMENT_TREE` /
/// `PARTITION_TREE` / `INTRA_MODE_TREE` (error-prone to re-derive by hand three times).
fn tree_path(tree: &[i32], leaf: i32) -> Vec<(usize, bool)> {
    fn search(tree: &[i32], idx: usize, leaf: i32, path: &mut Vec<(usize, bool)>) -> bool {
        let node = idx >> 1;
        for bit in [false, true] {
            let next = tree[idx + bit as usize];
            path.push((node, bit));
            let found = if next <= 0 {
                -next == leaf
            } else {
                search(tree, next as usize, leaf, path)
            };
            if found {
                return true;
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::new();
    assert!(search(tree, 0, leaf, &mut path), "leaf {leaf} not in tree");
    path
}

fn encode_tree(enc: &mut BoolEncoder, tree: &[i32], leaf: u8, probs: impl Fn(usize) -> u8) {
    for (node, bit) in tree_path(tree, leaf as i32) {
        enc.write_bool(bit, probs(node));
    }
}

fn write_no_update(enc: &mut BoolEncoder, count: usize) {
    // `diff_update_prob`/`update_mv_prob` (src/compressed_header.rs) both gate their update
    // on a single `B(252)` bit; writing `false` at p=252 keeps every table at its default.
    // CAUTION: a wrong `count` at a call site is invisible to every test in this file -- too few
    // writes are absorbed (the bool decoder reads the zero padding as more `false` bits), too
    // many are discarded by `exit_bool` -- so verify counts against
    // `parse_compressed_header_ex`'s read order when editing; green tests do not prove them.
    for _ in 0..count {
        enc.write_bool(false, 252);
    }
}

// ===========================================================================================
// Segmentation params (uncompressed-header `segmentation_params()`, spec 6.2.11).
// ===========================================================================================

/// Feature widths/signedness (spec 6.2.11 table); private consts in `src/header.rs`
/// (`SEGMENTATION_FEATURE_BITS`/`_SIGNED`), so re-declared here from the cited spec table.
const FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 2, 0];
const FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];

#[derive(Clone)]
struct SegSpec {
    enabled: bool,
    update_map: bool,
    tree_probs: [u8; 7],
    update_data: bool,
    abs_or_delta_update: bool,
    feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl SegSpec {
    fn disabled() -> Self {
        Self {
            enabled: false,
            update_map: false,
            tree_probs: [128; 7],
            update_data: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }

    /// Common base for every scenario below: segmentation on, an explicitly coded segment map
    /// (`update_map`), feature data present (`update_data`); `tree_probs` stays at `disabled()`'s
    /// `[128; 7]`, which the tile encoders mirror.
    fn enabled() -> Self {
        Self {
            enabled: true,
            update_map: true,
            update_data: true,
            ..Self::disabled()
        }
    }

    /// Writes `segmentation_params()`. `temporal_update` is always signaled `false`: none of
    /// this file's tests need temporal seg-id prediction, so `pred_prob` is never read.
    fn write(&self, w: &mut BitWriter) {
        w.push_flag(self.enabled);
        if !self.enabled {
            return;
        }
        w.push_flag(self.update_map);
        if self.update_map {
            for &p in &self.tree_probs {
                w.push_flag(true); // read_prob: coded (always write the 8-bit value explicitly)
                w.push_bits(p as u32, 8);
            }
            w.push_flag(false); // temporal_update = 0 -> pred_prob not read
        }
        w.push_flag(self.update_data);
        if self.update_data {
            w.push_flag(self.abs_or_delta_update);
            for seg in 0..MAX_SEGMENTS {
                for level in 0..SEG_LVL_MAX {
                    let enabled = self.feature_enabled[seg][level];
                    w.push_flag(enabled);
                    if enabled {
                        let bits = FEATURE_BITS[level];
                        let val = self.feature_data[seg][level];
                        if bits > 0 {
                            w.push_bits(val.unsigned_abs(), bits);
                        }
                        if FEATURE_SIGNED[level] {
                            w.push_flag(val < 0);
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================================
// Uncompressed header builders. Both mirror `src/header.rs`'s
// `#[cfg(test)] fn build_minimal_keyframe_header`/`build_minimal_inter_frame_header`, extended
// with segmentation/loop-filter-level parameters. `refresh_frame_context = false` +
// `frame_parallel_decoding_mode = true` on every frame here sidesteps backward probability
// adaptation entirely (`Decoder::decode_one_frame`'s `refresh_probs`, src/lib.rs) -- with only
// two frames ever decoded and the first always a key frame (which resets every frame context
// slot to defaults regardless), the adapted values are never actually consulted, so there's no
// need to hand-verify adaptation-with-all-zero-counts arithmetic here.
// ===========================================================================================

fn build_keyframe_header(
    loop_filter_level: u8,
    segmentation: &SegSpec,
    header_size_in_bytes: u16,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_bits(2, 2); // frame_marker
    w.push_bits(0, 1); // profile_low_bit
    w.push_bits(0, 1); // profile_high_bit -> profile 0
    w.push_flag(false); // show_existing_frame
    w.push_bits(0, 1); // frame_type = KEY_FRAME
    w.push_flag(true); // show_frame
    w.push_flag(false); // error_resilient_mode
    w.push_bits(0x49, 8);
    w.push_bits(0x83, 8);
    w.push_bits(0x42, 8);
    w.push_bits(0, 3); // color_space = CS_UNKNOWN
    w.push_flag(false); // color_range
    w.push_bits(WIDTH - 1, 16);
    w.push_bits(HEIGHT - 1, 16);
    w.push_flag(false); // render_size same as frame size
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
    w.push_bits(loop_filter_level as u32, 6);
    w.push_bits(0, 3); // sharpness
    w.push_flag(false); // loop_filter_delta_enabled
    w.push_bits(0, 8); // base_q_idx = 0 -> lossless
    w.push_flag(false); // delta_q_y_dc coded?
    w.push_flag(false); // delta_q_uv_dc coded?
    w.push_flag(false); // delta_q_uv_ac coded?
    segmentation.write(&mut w);
    w.push_bits(0, 1); // tile_rows_log2 (Sb64Cols=1 for a 16x16 frame -> tile_cols loop never runs)
    w.push_bits(header_size_in_bytes as u32, 16);
    w.finish()
}

fn build_inter_header(
    ref_frame_idx: [u8; 3],
    loop_filter_level: u8,
    segmentation: &SegSpec,
    header_size_in_bytes: u16,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_bits(2, 2);
    w.push_bits(0, 1);
    w.push_bits(0, 1);
    w.push_flag(false); // show_existing_frame
    w.push_bits(1, 1); // frame_type = NON_KEY_FRAME
    w.push_flag(true); // show_frame = 1 -> intra_only not read
    w.push_flag(false); // error_resilient_mode
    w.push_bits(0, 2); // reset_frame_context (irrelevant: this decoder only ever runs 2 frames)
    w.push_bits(0, 8); // refresh_frame_flags = 0: this frame is never used as a future reference
    for &idx in &ref_frame_idx {
        w.push_bits(idx as u32, 3);
        // ref_frame_sign_bias kept equal (false) across LAST/GOLDEN/ALTREF so
        // frame_reference_mode() takes its compound_reference_allowed == false shortcut
        // (SINGLE_REFERENCE, zero bits) -- see build_inter_compressed_header.
        w.push_flag(false);
    }
    w.push_flag(true); // frame_size_with_refs: inherit ref_frame_idx[0]'s slot size
    w.push_flag(false); // render_size same as frame size
    w.push_flag(false); // allow_high_precision_mv
    w.push_flag(false); // interpolation_filter: not switchable
    w.push_bits(0, 2); // -> EIGHTTAP_SMOOTH (LITERAL_TO_TYPE[0]); never read per-block
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx_raw
    w.push_bits(loop_filter_level as u32, 6);
    w.push_bits(0, 3);
    w.push_flag(false);
    w.push_bits(0, 8); // base_q_idx = 0 -> lossless
    w.push_flag(false);
    w.push_flag(false);
    w.push_flag(false);
    segmentation.write(&mut w);
    w.push_bits(0, 1);
    w.push_bits(header_size_in_bytes as u32, 16);
    w.finish()
}

/// Builds an `intra_only` (spec 6.2) non-key, hidden (`show_frame = 0`) frame's uncompressed
/// header, used only to plant distinct content in a single DPB slot (`refresh_frame_flags`)
/// without disturbing the others. Front half mirrors `src/header.rs::parse_uncompressed_header`'s
/// `intra_only` branch (verified at lines 618-650); the tail from `refresh_frame_context` onward
/// is byte-for-bit identical to `build_keyframe_header`'s (both key frames and intra_only frames
/// have `frame_is_intra = true` and share the same post-frame-size parsing code, lines 680-705).
fn build_intra_only_header(
    refresh_frame_flags: u8,
    segmentation: &SegSpec,
    header_size_in_bytes: u16,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.push_bits(2, 2); // frame_marker
    w.push_bits(0, 1); // profile_low_bit
    w.push_bits(0, 1); // profile_high_bit -> profile 0
    w.push_flag(false); // show_existing_frame
    w.push_bits(1, 1); // frame_type = NON_KEY_FRAME
    w.push_flag(false); // show_frame = 0 -> intra_only is read
    w.push_flag(false); // error_resilient_mode
    w.push_flag(true); // intra_only
    w.push_bits(0, 2); // reset_frame_context
    w.push_bits(0x49, 8);
    w.push_bits(0x83, 8);
    w.push_bits(0x42, 8);
    // profile == 0 -> color_config is not read (defaults used); see header.rs:633-643.
    w.push_bits(refresh_frame_flags as u32, 8);
    w.push_bits(WIDTH - 1, 16);
    w.push_bits(HEIGHT - 1, 16);
    w.push_flag(false); // render_size same as frame size
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
    w.push_bits(0, 6); // loop_filter_level
    w.push_bits(0, 3); // sharpness
    w.push_flag(false); // loop_filter_delta_enabled
    w.push_bits(0, 8); // base_q_idx = 0 -> lossless
    w.push_flag(false); // delta_q_y_dc coded?
    w.push_flag(false); // delta_q_uv_dc coded?
    w.push_flag(false); // delta_q_uv_ac coded?
    segmentation.write(&mut w);
    w.push_bits(0, 1); // tile_rows_log2
    w.push_bits(header_size_in_bytes as u32, 16);
    w.finish()
}

// ===========================================================================================
// Compressed header builders (bool-coded; spec 6.3). Both lossless, so `read_tx_mode` reads no
// bits at all (`ONLY_4X4` forced) -- see `src/compressed_header.rs`'s own
// `lossless_frame_forces_only_4x4_and_reads_no_extra_bit` test, mirrored here.
// ===========================================================================================

fn build_keyframe_compressed_header() -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    enc.write_bool(false, 128); // read_coef_probs: txSz=TX_4X4 (only iteration), update_probs=0
    write_no_update(&mut enc, 3); // read_skip_prob
    enc.finish()
}

/// Mirrors `parse_compressed_header_ex`'s read order (`src/compressed_header.rs:533-573`)
/// exactly. Every table update is declined (`write_no_update`), so this frame's
/// `CompressedHeaderProbs` stay at `CompressedHeaderProbs::default()` -- which is also what
/// `encode_inter_tile_forced` assumes when it needs a probability value at all (partition).
fn build_inter_compressed_header() -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    enc.write_bool(false, 128); // read_coef_probs
    write_no_update(&mut enc, 3); // read_skip_prob
    write_no_update(&mut enc, 7 * 3); // read_inter_mode_probs
                                      // interpolation_filter != SWITCHABLE -> read_interp_filter_probs not called
    write_no_update(&mut enc, 4); // read_is_inter_probs
                                  // frame_reference_mode(): ref_frame_sign_bias all equal -> compound_reference_allowed ==
                                  // false -> SINGLE_REFERENCE chosen without reading a single bit.
    write_no_update(&mut enc, 5 * 2); // frame_reference_mode_probs: single_ref_prob (comp_mode/comp_ref skipped)
    write_no_update(&mut enc, 4 * 9); // read_y_mode_probs
    write_no_update(&mut enc, 16 * 3); // read_partition_probs
    write_no_update(&mut enc, 3); // mv_probs: mv_joint_probs
    for _ in 0..2 {
        write_no_update(&mut enc, 1 + 10 + 1 + 10); // mv_sign, mv_class(10), mv_class0_bit, mv_bits(10)
    }
    for _ in 0..2 {
        write_no_update(&mut enc, 2 * 3 + 3); // mv_class0_fr[2][3], mv_fr[3]
    }
    // allow_high_precision_mv == false -> mv_class0_hp/mv_hp not read
    enc.finish()
}

// ===========================================================================================
// Tile data builders. Both build a fixed 16x16 (2x2 8x8-block) partition shape: `decode_tile`
// starts at BLOCK_64X64 and BLOCK_32X32, both of which auto-resolve to PARTITION_SPLIT with
// *no* bits read (MiRows == MiCols == 2, so `half_block8x8` always exceeds them -- hasRows/
// hasCols are both false); the BLOCK_16X16 level is the first with hasRows == hasCols == true,
// so it's the first real partition read (ctx=4, since above/left partition context start at
// all-zero), and each of the resulting four BLOCK_8X8 leaves reads a second, always-ctx-0
// partition bit (hand-verified: the partition-context bit this ctx depends on, bit index 3,
// is never set by an 8x8-sized update -- see `src/tile.rs::read_partition`/`decode_partition`).
// ===========================================================================================

const BLOCK_POSITIONS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

/// One block's plan for `encode_keyframe_tile`: `segment_id` is `Some` only when the frame's
/// segmentation has `update_map == true` (else `intra_segment_id` reads no bits at all).
/// Assumes `SEG_LVL_SKIP` is never active for these segments -- `skip` is always encoded as a
/// real bit (`read_skip`'s forced-without-a-bit path is only exercised by
/// `encode_inter_tile_forced`, never by this keyframe-only helper).
struct KeyBlock {
    segment_id: Option<u8>,
    skip: bool,
    y_mode: u8,
    uv_mode: u8,
}

/// Every block in this file's scenarios uses `skip = true` (no residual tokens -- the whole
/// design premise) and `uv_mode = DC_PRED`; only the segment id and y_mode vary.
fn kb(segment_id: Option<u8>, y_mode: u8) -> KeyBlock {
    KeyBlock {
        segment_id,
        skip: true,
        y_mode,
        uv_mode: DC_PRED,
    }
}

fn encode_keyframe_tile(blocks: [KeyBlock; 4], seg_tree_probs: [u8; 7]) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    encode_tree(&mut enc, &PARTITION_TREE, PARTITION_SPLIT, |n| {
        KF_PARTITION_PROBS[4][n]
    });

    let mut prev_skip = [[false; 2]; 2];
    let mut prev_mode = [[DC_PRED; 2]; 2];
    for (i, &(row, col)) in BLOCK_POSITIONS.iter().enumerate() {
        let avail_u = row > 0;
        let avail_l = col > 0;

        encode_tree(&mut enc, &PARTITION_TREE, PARTITION_NONE, |n| {
            KF_PARTITION_PROBS[0][n]
        });

        if let Some(seg) = blocks[i].segment_id {
            encode_tree(&mut enc, &SEGMENT_TREE, seg, |n| seg_tree_probs[n]);
        }

        let mut skip_ctx = 0usize;
        if avail_u && prev_skip[row - 1][col] {
            skip_ctx += 1;
        }
        if avail_l && prev_skip[row][col - 1] {
            skip_ctx += 1;
        }
        enc.write_bool(blocks[i].skip, DEFAULT_SKIP_PROB[skip_ctx]);

        let above_mode = if avail_u {
            prev_mode[row - 1][col]
        } else {
            DC_PRED
        };
        let left_mode = if avail_l {
            prev_mode[row][col - 1]
        } else {
            DC_PRED
        };
        let y_mode = blocks[i].y_mode;
        encode_tree(&mut enc, &INTRA_MODE_TREE, y_mode, |n| {
            KF_Y_MODE_PROBS[above_mode as usize][left_mode as usize][n]
        });
        encode_tree(&mut enc, &INTRA_MODE_TREE, blocks[i].uv_mode, |n| {
            KF_UV_MODE_PROBS[y_mode as usize][n]
        });

        prev_skip[row][col] = blocks[i].skip;
        prev_mode[row][col] = y_mode;
    }
    enc.finish()
}

/// Encodes the inter frame's tile data for the SEG_LVL_SKIP + SEG_LVL_REF_FRAME test: every
/// block maps (via `segment_id_per_block`) to a segment with both features active, so per
/// `src/tile.rs`, `read_skip`/`read_is_inter`/`read_ref_frames` all return forced values
/// without reading a bit, `inter_block_mode_info` forces `y_mode = ZEROMV` without reading
/// `inter_mode`, and `assign_mv` for `ZEROMV` reads no MV bits -- so `segment_id` (and the
/// partition shape) are the *only* bits this tile data contains.
fn encode_inter_tile_forced(segment_id_per_block: [u8; 4], seg_tree_probs: [u8; 7]) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    encode_tree(&mut enc, &PARTITION_TREE, PARTITION_SPLIT, |n| {
        DEFAULT_PARTITION_PROBS[4][n]
    });
    for &seg in &segment_id_per_block {
        encode_tree(&mut enc, &PARTITION_TREE, PARTITION_NONE, |n| {
            DEFAULT_PARTITION_PROBS[0][n]
        });
        encode_tree(&mut enc, &SEGMENT_TREE, seg, |n| seg_tree_probs[n]);
    }
    enc.finish()
}

/// The uncompressed header's `header_size_in_bytes` field is 16 bits (spec 6.2); an oversized
/// compressed header must fail loudly here, not wrap into a wrong tile-data offset.
fn header_size(compressed: &[u8]) -> u16 {
    u16::try_from(compressed.len()).expect("compressed header exceeds the 16-bit size field")
}

fn assemble_frame(header: Vec<u8>, compressed: Vec<u8>, tile: Vec<u8>) -> Vec<u8> {
    let mut frame = header;
    frame.extend_from_slice(&compressed);
    frame.extend_from_slice(&tile);
    frame
}

/// Decodes one chunk that must contain exactly one constituent frame (true for every
/// synthetic stream in this file -- none of them pack superframes).
fn decode_single(decoder: &mut Decoder, chunk: &[u8], what: &str) -> DecodedFrame {
    let mut decoded = decoder
        .decode_frame(chunk)
        .unwrap_or_else(|e| panic!("{what} should decode: {e:?}"));
    assert_eq!(
        decoded.len(),
        1,
        "{what}: expected exactly one constituent frame"
    );
    decoded.pop().unwrap()
}

// ===========================================================================================
// Test 1: SEG_LVL_SKIP + SEG_LVL_REF_FRAME.
//
// A non-flat key frame followed by an inter frame whose one segment forces SEG_LVL_SKIP +
// SEG_LVL_REF_FRAME=LAST for every block. Two #[test] fns share this scenario (one per
// feature) since both are exercised by the same forced-decode bitstream and the same
// pixel-copy oracle; splitting the assertions still gives each feature its own named,
// independently-failing test per the task's "three test function names" ask.
//
// Row 0 (V_PRED, no "above" neighbor) predicts flat 127; row 1 (H_PRED, no "left" neighbor on
// its first column, and H_PRED never reads "above" so it never inherits row 0's value) predicts
// flat 129 -- the same real, hand-verified 127/129 split used by the SEG_LVL_ALT_L test below.
// (An earlier draft assigned all four blocks *different* modes -- V/H/TM/D45/DC_PRED -- meaning
// to produce a four-way-varied image, but TM_PRED/DC_PRED average the real above/left neighbor
// values once a neighbor is available, and every block past the first has a real neighbor, so
// the whole frame collapsed back to a uniform 127; the V_PRED/H_PRED split is the smallest
// change that gives a real, persistent, exactly-known-value edge that skip=1 alone can't erase.)
// ===========================================================================================

/// Builds the ordered raw VP9 frame byte-streams for the SEG_LVL_SKIP + SEG_LVL_REF_FRAME
/// scenario: `[keyframe, inter]`. Shared by `decode_skip_ref_frame_scenario` (the existing
/// tests) and the external-cross-decode dump harness, so both decode exactly the same bytes.
fn build_skip_ref_frames() -> Vec<Vec<u8>> {
    let keyframe_compressed = build_keyframe_compressed_header();
    let keyframe_header =
        build_keyframe_header(0, &SegSpec::disabled(), header_size(&keyframe_compressed));
    let keyframe_tile = encode_keyframe_tile(
        [
            kb(None, V_PRED),
            kb(None, V_PRED),
            kb(None, H_PRED),
            kb(None, H_PRED),
        ],
        [128; 7],
    );
    let keyframe_bytes = assemble_frame(keyframe_header, keyframe_compressed, keyframe_tile);

    let mut seg = SegSpec::enabled();
    seg.feature_enabled[0][SEG_LVL_SKIP] = true;
    seg.feature_enabled[0][SEG_LVL_REF_FRAME] = true;
    seg.feature_data[0][SEG_LVL_REF_FRAME] = LAST_FRAME as i32;

    let inter_compressed = build_inter_compressed_header();
    let inter_header = build_inter_header([0, 0, 0], 0, &seg, header_size(&inter_compressed));
    let inter_tile = encode_inter_tile_forced([0, 0, 0, 0], seg.tree_probs);
    let inter_bytes = assemble_frame(inter_header, inter_compressed, inter_tile);

    vec![keyframe_bytes, inter_bytes]
}

fn decode_skip_ref_frame_scenario() -> (vp9dec::Frame, vp9dec::Frame, vp9dec::FrameDecodeInfo) {
    let frames = build_skip_ref_frames();

    let mut decoder = Decoder::new();
    let key_frame = decode_single(&mut decoder, &frames[0], "keyframe")
        .frame
        .expect("keyframe has show_frame = 1");
    let inter = decode_single(&mut decoder, &frames[1], "inter frame");
    let info = inter.info.expect("info recorded for a newly decoded frame");
    let inter_frame = inter.frame.expect("inter frame has show_frame = 1");
    (key_frame, inter_frame, info)
}

#[test]
fn seg_lvl_skip_forces_exact_copy_without_residual_or_mv_bits() {
    let (key_frame, inter_frame, info) = decode_skip_ref_frame_scenario();

    assert!(info.seg_features_active[SEG_LVL_SKIP]);
    // The key frame really is non-flat (see the module comment above this scenario for why
    // V_PRED/H_PRED rather than 4 distinct modes): confirms the pixel-exact check below isn't
    // vacuously comparing two uniform-grey frames.
    for x in 0..WIDTH as usize {
        assert_eq!(key_frame.y[x], 127, "row0");
        assert_eq!(key_frame.y[8 * WIDTH as usize + x], 129, "row8");
    }
    // Proven: `read_skip` took the seg_feature_active(SEG_LVL_SKIP) forced-true path without
    // consuming a bit, `inter_block_mode_info` forced y_mode=ZEROMV without reading
    // `inter_mode`, and `assign_mv` read no MV bits for it -- the supplied tile data contains
    // none of those bits, so any other code path would either error out (bool decoder running
    // past the end of a byte-padded, all-forced-value stream can desync silently, not just
    // panic) or reconstruct different pixels. A desync would very likely show up as a mismatch
    // somewhere in this non-flat image.
    // Not proven: which specific reference was used (see the SEG_LVL_REF_FRAME test).
    assert_eq!(key_frame, inter_frame);
}

#[test]
fn seg_lvl_ref_frame_selects_the_forced_reference_without_reading_bits() {
    let (key_frame, inter_frame, info) = decode_skip_ref_frame_scenario();

    assert!(info.seg_features_active[SEG_LVL_REF_FRAME]);
    // Proven: `read_is_inter`/`read_ref_frames` both took their seg_feature_active(
    // SEG_LVL_REF_FRAME) forced path (is_inter=true, ref_frame=[LAST_FRAME, NONE]) without
    // reading a bit or a compound-reference-mode selector, and motion compensation with
    // ZEROMV against that forced LAST reference reproduces the key frame exactly.
    // Not proven here (every DPB slot holds the same key frame, so copying "LAST" is
    // indistinguishable from always copying "the" single reference regardless of FeatureData):
    // that REF_FRAME steers to a *particular* slot. See
    // `seg_lvl_ref_frame_steers_to_the_specific_slot_not_just_last` below, which plants
    // different content in LAST's and GOLDEN's slots and confirms FeatureData = GOLDEN_FRAME
    // actually resolves to GOLDEN's slot.
    assert_eq!(key_frame, inter_frame);
}

// ===========================================================================================
// Test 1b: SEG_LVL_REF_FRAME steers to a *specific* DPB slot, not just "the" reference.
//
// Three frames: (1) a key frame, content A = flat 127, refreshes all 8 slots; (2) a hidden
// (show_frame=0) intra_only frame, content B = flat 129, refreshes ONLY physical slot 1 (so
// slot 0 keeps A, slot 1 becomes B); (3) an inter frame with ref_frame_idx = [0, 1, 2]
// (LAST->slot0=A, GOLDEN->slot1=B) and SEG_LVL_REF_FRAME's feature_data = GOLDEN_FRAME. A
// correct decode copies GOLDEN's content (B = 129); a decoder that read FeatureData but
// resolved the wrong slot, or one that ignored FeatureData and always used LAST, would instead
// reproduce A = 127 -- so the two contents being genuinely different pixel values (not just
// different frame objects) is what makes this discriminating. A same-decoder companion decode
// steered to LAST (asserted flat 127) additionally pins that slot 0 still holds A when the
// discriminator runs -- see the end of the test.
// ===========================================================================================

/// Builds the ordered raw VP9 frame byte-streams for the SEG_LVL_REF_FRAME slot-steering
/// scenario: `[keyframe, intra_only_hidden, inter]`. Shared by
/// `seg_lvl_ref_frame_steers_to_the_specific_slot_not_just_last` and the external-cross-decode
/// dump harness, so both decode exactly the same bytes.
fn build_steering_frames() -> Vec<Vec<u8>> {
    let keyframe_compressed = build_keyframe_compressed_header();
    let keyframe_header =
        build_keyframe_header(0, &SegSpec::disabled(), header_size(&keyframe_compressed));
    let keyframe_tile = encode_keyframe_tile(
        [
            kb(None, V_PRED),
            kb(None, V_PRED),
            kb(None, V_PRED),
            kb(None, V_PRED),
        ],
        [128; 7],
    );
    let keyframe_bytes = assemble_frame(keyframe_header, keyframe_compressed, keyframe_tile);

    // intra_only, hidden: frame_is_intra = true, so this tile is encoded exactly like a key
    // frame's (same KF_* mode probs, same compressed-header shape) -- see build_intra_only_header.
    let intra_only_compressed = build_keyframe_compressed_header();
    let intra_only_header = build_intra_only_header(
        0x02, // refresh ONLY physical slot 1 -- slot 0 (and the rest) keep the key frame's A.
        &SegSpec::disabled(),
        header_size(&intra_only_compressed),
    );
    let intra_only_tile = encode_keyframe_tile(
        [
            kb(None, H_PRED),
            kb(None, H_PRED),
            kb(None, H_PRED),
            kb(None, H_PRED),
        ],
        [128; 7],
    );
    let intra_only_bytes =
        assemble_frame(intra_only_header, intra_only_compressed, intra_only_tile);

    let mut seg = SegSpec::enabled();
    seg.feature_enabled[0][SEG_LVL_SKIP] = true;
    seg.feature_enabled[0][SEG_LVL_REF_FRAME] = true;
    seg.feature_data[0][SEG_LVL_REF_FRAME] = GOLDEN_FRAME as i32;

    let inter_compressed = build_inter_compressed_header();
    // LAST->slot0 (A), GOLDEN->slot1 (B), ALTREF->slot2 (unused, still A from the key frame).
    let inter_header = build_inter_header([0, 1, 2], 0, &seg, header_size(&inter_compressed));
    let inter_tile = encode_inter_tile_forced([0, 0, 0, 0], seg.tree_probs);
    let inter_bytes = assemble_frame(inter_header, inter_compressed, inter_tile);

    vec![keyframe_bytes, intra_only_bytes, inter_bytes]
}

#[test]
fn seg_lvl_ref_frame_steers_to_the_specific_slot_not_just_last() {
    let frames = build_steering_frames();

    let mut decoder = Decoder::new();
    let key_frame = decode_single(&mut decoder, &frames[0], "keyframe")
        .frame
        .expect("keyframe has show_frame = 1");
    let hidden = decode_single(&mut decoder, &frames[1], "intra_only frame");
    assert!(
        hidden.frame.is_none(),
        "intra_only frame has show_frame = 0 -> no visible output"
    );
    let inter = decode_single(&mut decoder, &frames[2], "inter frame");
    let info = inter.info.expect("info recorded for a newly decoded frame");
    let inter_frame = inter.frame.expect("inter frame has show_frame = 1");

    assert!(info.seg_features_active[SEG_LVL_REF_FRAME]);
    // Sanity: content A (the key frame, held by slot 0/LAST) really is flat 127 everywhere --
    // confirms the two slots hold genuinely different content, so the discriminator below isn't
    // vacuously comparing equal pixels.
    for (i, &px) in key_frame.y.iter().enumerate() {
        assert_eq!(
            px, 127,
            "key frame (slot 0/LAST) must be flat A; Y pixel {i}"
        );
    }
    // Discriminator: FeatureData = GOLDEN_FRAME must resolve to ref_frame_idx[GOLDEN]'s slot
    // (physical slot 1 = B = 129), not to LAST's slot (physical slot 0 = A = 127), in *every*
    // block -- a decoder that ignored FeatureData, or resolved the wrong slot for only some
    // blocks (e.g. a ctx- or position-dependent fallback), would leave 127 somewhere.
    for (i, &px) in inter_frame.y.iter().enumerate() {
        assert_eq!(
            px, 129,
            "SEG_LVL_REF_FRAME=GOLDEN must copy slot 1's content (B) everywhere; Y pixel {i}"
        );
    }

    // Companion probe: the same decoder, steered to LAST instead, must reproduce A. This pins
    // the discriminator's premise in-test -- slot 0 still holds A after the intra_only frame's
    // slot-1-only refresh -- so the GOLDEN=129 check above cannot be satisfied by an
    // over-refresh regression that put B in every slot. (Deliberately built here and not in
    // build_steering_frames, so the dumped/cross-decoded stream stays as recorded in the notes.)
    let mut seg = SegSpec::enabled();
    seg.feature_enabled[0][SEG_LVL_SKIP] = true;
    seg.feature_enabled[0][SEG_LVL_REF_FRAME] = true;
    seg.feature_data[0][SEG_LVL_REF_FRAME] = LAST_FRAME as i32;
    let compressed = build_inter_compressed_header();
    let header = build_inter_header([0, 1, 2], 0, &seg, header_size(&compressed));
    let tile = encode_inter_tile_forced([0, 0, 0, 0], seg.tree_probs);
    let last_steered = decode_single(
        &mut decoder,
        &assemble_frame(header, compressed, tile),
        "LAST-steered companion frame",
    )
    .frame
    .expect("companion has show_frame = 1");
    for (i, &px) in last_steered.y.iter().enumerate() {
        assert_eq!(
            px, 127,
            "LAST-steered companion must copy slot 0's content (A); Y pixel {i}"
        );
    }
}

// ===========================================================================================
// Test 2: SEG_LVL_ALT_L.
//
// A key frame with 2 segments: row 0 (blocks (0,0)/(0,1), segment 0, no ALT_L) is V_PRED with
// no "above" neighbor, which per `src/predict.rs::predict_intra`'s DC_PRED/V_PRED unavailable-
// neighbor fallback (`base - 1` = 127 for 8-bit) predicts flat 127 everywhere; row 1 (blocks
// (1,0)/(1,1), segment 1, ALT_L under test) is H_PRED with no "left" neighbor on its first
// column (`base + 1` = 129), and since H_PRED never reads "above", it comes out flat 129
// regardless of segment. So there is an exact, hand-computable jump of 2 (127 -> 129) at the
// horizontal block edge y=8, entirely inside segment 1 (the loop filter looks up the filter
// level from the position's *own* block -- `src/loop_filter.rs::superblock_loop_filter`'s
// `loop_row`/`mi.segment_id` -- and y=8 belongs to row 1's blocks).
// ===========================================================================================

/// Builds the ordered raw VP9 frame byte-stream (a single key frame) for the fixed 2-segment
/// scenario with segment 1's `SEG_LVL_ALT_L` feature data set to `alt_l_level` (an absolute
/// override, `abs_or_delta_update = true`). Shared by `decode_alt_l_frame` (the existing test)
/// and the external-cross-decode dump harness, so both decode exactly the same bytes.
fn build_alt_l_frames(alt_l_level: i32) -> Vec<Vec<u8>> {
    let mut seg = SegSpec::enabled();
    seg.abs_or_delta_update = true;
    seg.feature_enabled[1][SEG_LVL_ALT_L] = true;
    seg.feature_data[1][SEG_LVL_ALT_L] = alt_l_level;

    let compressed = build_keyframe_compressed_header();
    // Base level 30 (nonzero, as the task asks) only ever applies to segment 0's own edges
    // (none of which this test examines): segment 1's `abs_or_delta_update = true` override
    // replaces the base level outright wherever segment 1 is looked up.
    let header = build_keyframe_header(30, &seg, header_size(&compressed));
    let tile = encode_keyframe_tile(
        [
            kb(Some(0), V_PRED),
            kb(Some(0), V_PRED),
            kb(Some(1), H_PRED),
            kb(Some(1), H_PRED),
        ],
        seg.tree_probs,
    );
    let frame_bytes = assemble_frame(header, compressed, tile);

    vec![frame_bytes]
}

/// Decodes the fixed 2-segment key frame with segment 1's `SEG_LVL_ALT_L` feature data set to
/// `alt_l_level` (an absolute override, `abs_or_delta_update = true`).
fn decode_alt_l_frame(alt_l_level: i32) -> vp9dec::Frame {
    let frames = build_alt_l_frames(alt_l_level);

    let mut decoder = Decoder::new();
    decode_single(&mut decoder, &frames[0], "alt_l key frame")
        .frame
        .expect("key frame has show_frame = 1")
}

#[test]
fn seg_lvl_alt_l_loop_filter_level_change_is_observable() {
    let unfiltered = decode_alt_l_frame(0);
    let filtered = decode_alt_l_frame(63);

    assert_ne!(unfiltered, filtered);

    // Hand-derived from spec 8.8.5.2 / `src/loop_filter.rs::narrow_filter` (the filter size
    // used here: lossless forces tx_size=TX_4X4 for every block, and this edge isn't a 32-pel
    // boundary, so `filter_size_process` picks TX_4X4 => narrow_filter, not the wide variant).
    // With p1=p0=127 (row 0) and q0=q1=129 (row 1): hevMask is false (both inner deltas are 0,
    // under any nonzero thresh), so filter = clamp(3*(qs0-ps0)) = clamp(3*2) = 6,
    // filter1 = (6+4)>>3 = 1, filter2 = (6+3)>>3 = 1, and since hevMask is false, p1/q1 are
    // also adjusted by round2(filter1, 1) = 1. All four narrow to 128. filterMask is true
    // throughout (all the "is this actually a big edge" checks stay comfortably under `limit`
    // at level 63 -- a jump of 2 is exactly the kind of mild block-boundary artifact this
    // filter targets, not a real edge it's designed to preserve), and lvl > 0 only when
    // `alt_l_level == 63` (segment 1's absolute override), so the level-0 decode leaves the
    // rows untouched instead.
    for x in 0..WIDTH as usize {
        assert_eq!(
            unfiltered.y[5 * WIDTH as usize + x],
            127,
            "row5 (untouched by either level)"
        );
        assert_eq!(
            unfiltered.y[7 * WIDTH as usize + x],
            127,
            "row7, alt_l=0: unfiltered"
        );
        assert_eq!(
            unfiltered.y[8 * WIDTH as usize + x],
            129,
            "row8, alt_l=0: unfiltered"
        );
        assert_eq!(
            unfiltered.y[10 * WIDTH as usize + x],
            129,
            "row10 (untouched by either level)"
        );

        assert_eq!(
            filtered.y[6 * WIDTH as usize + x],
            128,
            "row6, alt_l=63: filtered"
        );
        assert_eq!(
            filtered.y[7 * WIDTH as usize + x],
            128,
            "row7, alt_l=63: filtered"
        );
        assert_eq!(
            filtered.y[8 * WIDTH as usize + x],
            128,
            "row8, alt_l=63: filtered"
        );
        assert_eq!(
            filtered.y[9 * WIDTH as usize + x],
            128,
            "row9, alt_l=63: filtered"
        );
        // The narrow filter's reach is p1..q1 (rows 6-9 at this edge); rows 5 and 10 anchor
        // that the level-63 decode touched *only* the edge (a wide/flat-filter mis-selection
        // regression would pull these toward 128 too).
        assert_eq!(
            filtered.y[5 * WIDTH as usize + x],
            127,
            "row5, alt_l=63: must stay untouched"
        );
        assert_eq!(
            filtered.y[10 * WIDTH as usize + x],
            129,
            "row10, alt_l=63: must stay untouched"
        );
    }
}

// ===========================================================================================
// External cross-decode dump harness (see docs/implementation-notes.md, 2026-07-14 entry).
//
// Writes the exact same byte-streams the tests above decode to disk as `.ivf`, plus this
// decoder's own output as raw I420 `.yuv`, so an external VP9 decoder (e.g. ffmpeg) can
// independently confirm this decoder's interpretation of SEG_LVL_ALT_L/REF_FRAME/SKIP.
// ===========================================================================================

/// The four synthetic scenarios plus the shown-frame count each must produce. The single source
/// of truth for both the dump harness and the ffmpeg cross-decode test, so the dumped set can
/// never silently diverge from the cross-checked set.
fn scenarios() -> [(&'static str, Vec<Vec<u8>>, usize); 4] {
    [
        ("skip_ref", build_skip_ref_frames(), 2),
        ("steering", build_steering_frames(), 2),
        ("alt_l_0", build_alt_l_frames(0), 1),
        ("alt_l_63", build_alt_l_frames(63), 1),
    ]
}

/// Env-gated: no-op unless `VP9DEC_DUMP_DIR` is set, so a plain `cargo test` run stays green
/// without needing an external decoder available. Set it to dump `.ivf` + `.our_i420.yuv` pairs
/// for the four synthetic scenarios for comparison against an external VP9 decoder.
#[test]
fn dump_synthetic_ivf_for_external_cross_decode() {
    let Ok(dir) = std::env::var("VP9DEC_DUMP_DIR") else {
        eprintln!(
            "[skip] VP9DEC_DUMP_DIR not set, skipping dump. To enable, set it to an output \
             directory (PowerShell: $env:VP9DEC_DUMP_DIR=\"<dir>\"; bash: VP9DEC_DUMP_DIR=<dir>) \
             and run: cargo test --test synthetic_seg_test \
             dump_synthetic_ivf_for_external_cross_decode -- --nocapture"
        );
        return;
    };
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).expect("create VP9DEC_DUMP_DIR");

    for (name, frames, expected_shown) in scenarios() {
        let ivf_path = dir.join(format!("{name}.ivf"));
        let yuv_path = dir.join(format!("{name}.our_i420.yuv"));
        // A panic between the two writes below must not leave a fresh .ivf beside a previous
        // run's .yuv -- the pair is the input to the documented manual external comparison.
        let _ = std::fs::remove_file(&ivf_path);
        let _ = std::fs::remove_file(&yuv_path);

        let ivf_bytes =
            vp9dec::ivf::write_ivf(b"VP90", WIDTH as u16, HEIGHT as u16, 30, 1, &frames);
        std::fs::write(&ivf_path, &ivf_bytes).expect("write .ivf");

        let mut decoder = Decoder::new();
        let mut yuv = Vec::new();
        let mut shown_count = 0usize;
        for frame in &frames {
            for df in decoder.decode_frame(frame).expect("frame should decode") {
                if let Some(decoded) = df.frame {
                    yuv.extend_from_slice(&decoded.y);
                    yuv.extend_from_slice(&decoded.u);
                    yuv.extend_from_slice(&decoded.v);
                    shown_count += 1;
                }
            }
        }
        assert_eq!(
            shown_count, expected_shown,
            "{name}: unexpected shown-frame count"
        );
        std::fs::write(&yuv_path, &yuv).expect("write .our_i420.yuv");

        eprintln!(
            "[dump] {}: {} bytes ({} frames total)",
            ivf_path.display(),
            ivf_bytes.len(),
            frames.len()
        );
        eprintln!(
            "[dump] {}: {} bytes ({shown_count} shown frame(s))",
            yuv_path.display(),
            yuv.len()
        );
    }
}

// ===========================================================================================
// Automatic external cross-decode test (see docs/implementation-notes.md, "External
// cross-decode result" entry). Turns the one-time manual ffmpeg comparison recorded there into
// a self-running, skip-if-absent test: it shells out to the ffmpeg *binary* via std::process
// only (no crate dependency -- the product and this test harness both stay zero-dependency) and
// probes for it first, so a plain `cargo test` on a machine without ffmpeg installed stays green,
// exactly like the conformance tests skip cleanly when vector files are absent.
// ===========================================================================================

/// Locates and probes an ffmpeg binary. Reads `VP9DEC_FFMPEG` (a full path) if set, else falls
/// back to `"ffmpeg"` on `PATH` -- never a hardcoded machine-specific path, since this test is
/// committed and must behave on any machine (present or absent ffmpeg alike). A set-but-unusable
/// `VP9DEC_FFMPEG` is an explicit misconfiguration and fails loudly instead of skipping -- a
/// silent [skip] on a passing run is invisible, so the user who set it would believe the
/// cross-decode ran.
fn probe_ffmpeg() -> Option<String> {
    let explicit = std::env::var("VP9DEC_FFMPEG").ok();
    let ffmpeg = explicit.clone().unwrap_or_else(|| "ffmpeg".to_string());
    let found = std::process::Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .is_ok_and(|out| out.status.success());
    assert!(
        found || explicit.is_none(),
        "VP9DEC_FFMPEG is set ({ffmpeg:?}) but does not run as ffmpeg"
    );
    found.then_some(ffmpeg)
}

/// The subset of `decoders` this ffmpeg build actually provides, per `-decoders` (LGPL/minimal
/// builds commonly ship without libvpx). Token-exact match on the decoder-name column: "vp9" is
/// a substring of "libvpx-vp9" and "vp9_qsv", so a contains() check would misreport.
fn available_decoders(ffmpeg: &str, decoders: [&'static str; 2]) -> Vec<&'static str> {
    let out = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-decoders"])
        .output()
        .expect("run ffmpeg -decoders");
    let listing = String::from_utf8_lossy(&out.stdout);
    decoders
        .into_iter()
        .filter(|d| {
            listing
                .lines()
                .any(|line| line.split_whitespace().nth(1) == Some(*d))
        })
        .collect()
}

#[test]
fn synthetic_streams_cross_decode_against_ffmpeg() {
    let Some(ffmpeg) = probe_ffmpeg() else {
        eprintln!(
            "[skip] ffmpeg not found (set VP9DEC_FFMPEG=<path> or put ffmpeg on PATH to run the \
             cross-decode)"
        );
        return;
    };
    // libvpx-vp9 is the reference implementation; vp9 is ffmpeg's own independent native
    // decoder. Run whichever this build provides: probing only `-version` and then asserting
    // decode success would hard-fail the suite on a libvpx-less build, breaking the
    // skip-cleanly contract this test promises.
    let decoders = available_decoders(&ffmpeg, ["libvpx-vp9", "vp9"]);
    if decoders.is_empty() {
        eprintln!("[skip] {ffmpeg:?} provides neither the libvpx-vp9 nor the vp9 decoder");
        return;
    }
    if decoders.len() < 2 {
        eprintln!(
            "[note] this ffmpeg build only provides {decoders:?}; cross-decoding with it alone"
        );
    }

    // 4:2:0 I420 frame size, derived from the same WIDTH/HEIGHT every bitstream builder uses.
    const I420_FRAME_SIZE: usize = (WIDTH * HEIGHT + 2 * (WIDTH / 2) * (HEIGHT / 2)) as usize;

    // Removed on drop rather than at the end of the function so a mid-test panic doesn't leave
    // the PID-named dir behind (Windows reuses PIDs, so a later run would adopt the stale dir).
    struct CleanOnDrop(std::path::PathBuf);
    impl Drop for CleanOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let tmp_dir = std::env::temp_dir().join(format!("vp9dec_xdecode_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let _cleanup = CleanOnDrop(tmp_dir.clone());

    for (name, frames, expected_shown) in scenarios() {
        let total = frames.len();
        let mut decoder = Decoder::new();
        let mut shown: Vec<(usize, Vec<u8>)> = Vec::new();
        for (idx, frame) in frames.iter().enumerate() {
            for df in decoder.decode_frame(frame).expect("frame should decode") {
                if let Some(decoded) = df.frame {
                    let mut i420 =
                        Vec::with_capacity(decoded.y.len() + decoded.u.len() + decoded.v.len());
                    i420.extend_from_slice(&decoded.y);
                    i420.extend_from_slice(&decoded.u);
                    i420.extend_from_slice(&decoded.v);
                    shown.push((idx, i420));
                }
            }
        }
        // Pin our own side too: a Decoder regression that changes which frames are shown must
        // fail here, not shrink the comparison set below and still report "OK".
        assert_eq!(
            shown.len(),
            expected_shown,
            "{name}: this decoder produced an unexpected shown-frame count"
        );

        let ivf_path = tmp_dir.join(format!("{name}.ivf"));
        std::fs::write(
            &ivf_path,
            vp9dec::ivf::write_ivf(b"VP90", WIDTH as u16, HEIGHT as u16, 30, 1, &frames),
        )
        .expect("write .ivf");

        for &decoder_name in &decoders {
            let out_path = tmp_dir.join(format!("{name}.{decoder_name}.yuv"));
            let output = std::process::Command::new(&ffmpeg)
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-c:v",
                    decoder_name,
                    "-i",
                ])
                .arg(&ivf_path)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&out_path)
                .output()
                .expect("run ffmpeg");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "ffmpeg -c:v {decoder_name} failed decoding {name}.ivf:\n{stderr}"
            );
            // At -loglevel error, any stderr output is a decode error ffmpeg chose to continue
            // past (without -xerror it exits 0 after per-frame errors) -- treat it as a failure
            // rather than compare against an output with silently dropped or corrupt frames.
            assert!(
                stderr.trim().is_empty(),
                "ffmpeg -c:v {decoder_name} reported errors decoding {name}.ivf:\n{stderr}"
            );

            let out = std::fs::read(&out_path).expect("read ffmpeg raw output");

            // ffmpeg emits one frame per constituent VP9 frame -- including the hidden
            // show_frame==0 frame this decoder deliberately never outputs (spec 8.9; recorded in
            // docs/implementation-notes.md's "External cross-decode result"). Requiring the
            // exact total (rather than accepting whichever of total/shown happens to match) is
            // load-bearing: steering's hidden frame and its inter frame are pixel-identical
            // (both flat 129), so a laxer length check could match a dropped final frame
            // against the hidden one and still pass.
            assert_eq!(
                out.len(),
                total * I420_FRAME_SIZE,
                "{name}/{decoder_name}: unexpected ffmpeg output size (total={total} constituent \
                 frames, frame_size={I420_FRAME_SIZE})"
            );
            for (idx, ours) in &shown {
                let idx = *idx;
                let expected = &out[idx * I420_FRAME_SIZE..(idx + 1) * I420_FRAME_SIZE];
                assert_eq!(
                    ours.as_slice(),
                    expected,
                    "{name}/{decoder_name}: constituent frame {idx} mismatch"
                );
            }

            eprintln!(
                "[xdecode] {name}/{decoder_name}: OK ({} shown frame(s) byte-identical)",
                shown.len()
            );
        }
    }
}
