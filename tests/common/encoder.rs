//! Synthetic VP9 bitstream *encoder* helpers: a tiny bool coder + uncompressed-header bit
//! writer (via `vp9dec::test_support`) plus the frame/tile builders used to hand-assemble just
//! enough of a VP9 stream to drive specific decode paths deliberately.
//!
//! Extracted from `tests/synthetic_seg_test.rs` (Wave 3 test-layer consolidation, 2026-07-16)
//! so it's reusable by future synthetic-vector test files (e.g. M4's reference-frame-scaling
//! coverage) without duplicating ~400 lines of encoder plumbing. `synthetic_seg_test.rs` keeps
//! its scenario builders and tests, importing this module's pieces.

use vp9dec::header::{MAX_SEGMENTS, SEG_LVL_MAX};
use vp9dec::prob_tables::{
    DC_PRED, DEFAULT_PARTITION_PROBS, DEFAULT_SKIP_PROB, INTRA_MODE_TREE, KF_PARTITION_PROBS,
    KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, PARTITION_NONE, PARTITION_SPLIT, PARTITION_TREE,
    SEGMENT_TREE,
};
use vp9dec::test_support::{BitWriter, BoolEncoder};

/// Default frame size for this module's fixed-size pieces (`build_intra_only_header`, the
/// tile builders' 2x2 8x8-block layout) and the seg/superframe tests built on them.
pub const WIDTH: u32 = 16;
pub const HEIGHT: u32 = 16;

/// Non-RGB key-frame format and tiling knobs for synthetic streams. The default reproduces
/// [`build_keyframe_header`]'s historical profile-0, 8-bit 4:2:0, single-tile encoding.
#[derive(Clone, Copy, Debug)]
pub struct KeyframeConfig {
    pub profile: u8,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub tile_cols_log2: u32,
}

impl Default for KeyframeConfig {
    fn default() -> Self {
        Self {
            profile: 0,
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            tile_cols_log2: 0,
        }
    }
}

/// Finds the `(node, bit)` sequence that `BoolDecoder::read_tree` (`src/bool_coder.rs`)
/// would read, in order, to arrive at `leaf`, by walking the tree structure the same way
/// `read_tree` does. Used instead of hand-transcribing bit paths for `SEGMENT_TREE` /
/// `PARTITION_TREE` / `INTRA_MODE_TREE` (error-prone to re-derive by hand three times).
pub fn tree_path(tree: &[i32], leaf: i32) -> Vec<(usize, bool)> {
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

pub fn encode_tree(enc: &mut BoolEncoder, tree: &[i32], leaf: u8, probs: impl Fn(usize) -> u8) {
    for (node, bit) in tree_path(tree, leaf as i32) {
        enc.write_bool(bit, probs(node));
    }
}

pub fn write_no_update(enc: &mut BoolEncoder, count: usize) {
    // `diff_update_prob`/`update_mv_prob` (src/compressed_header.rs) both gate their update
    // on a single `B(252)` bit; writing `false` at p=252 keeps every table at its default.
    // CAUTION: a wrong `count` at a call site is invisible to every test in this file -- too few
    // writes are absorbed (the bool decoder reads the zero padding as more `false` bits), too
    // many are discarded by `exit_bool` -- so verify counts against
    // `parse_compressed_header`'s read order (`src/compressed_header.rs`) when editing; green
    // tests do not prove them.
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
pub struct SegSpec {
    pub enabled: bool,
    pub update_map: bool,
    pub tree_probs: [u8; 7],
    pub update_data: bool,
    pub abs_or_delta_update: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl SegSpec {
    pub fn disabled() -> Self {
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
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            update_map: true,
            update_data: true,
            ..Self::disabled()
        }
    }

    /// Writes `segmentation_params()`. `temporal_update` is always signaled `false`: none of
    /// this file's tests need temporal seg-id prediction, so `pred_prob` is never read.
    pub fn write(&self, w: &mut BitWriter) {
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
// with frame-size/segmentation/loop-filter-level parameters. `refresh_frame_context = false` +
// `frame_parallel_decoding_mode = true` on every frame here sidesteps backward probability
// adaptation entirely (`Decoder::decode_one_frame`'s `refresh_probs`, src/lib.rs) -- with only
// two frames ever decoded and the first always a key frame (which resets every frame context
// slot to defaults regardless), the adapted values are never actually consulted, so there's no
// need to hand-verify adaptation-with-all-zero-counts arithmetic here.
// ===========================================================================================

pub fn build_keyframe_header(
    width: u32,
    height: u32,
    loop_filter_level: u8,
    loop_filter_delta_enabled: bool,
    segmentation: &SegSpec,
    header_size_in_bytes: u16,
) -> Vec<u8> {
    build_keyframe_header_with_config(
        width,
        height,
        loop_filter_level,
        loop_filter_delta_enabled,
        segmentation,
        header_size_in_bytes,
        KeyframeConfig::default(),
    )
}

/// Generalized form of [`build_keyframe_header`] for synthetic profile/bit-depth/subsampling
/// and tile-column coverage. It deliberately emits a non-RGB color config and one tile row,
/// which are the only variants currently needed by the synthetic integration tests.
pub fn build_keyframe_header_with_config(
    width: u32,
    height: u32,
    loop_filter_level: u8,
    loop_filter_delta_enabled: bool,
    segmentation: &SegSpec,
    header_size_in_bytes: u16,
    config: KeyframeConfig,
) -> Vec<u8> {
    assert!(width > 0 && height > 0, "frame dimensions must be nonzero");
    assert!(config.profile <= 3, "VP9 profile must be in 0..=3");
    if config.profile >= 2 {
        assert!(
            config.bit_depth == 10 || config.bit_depth == 12,
            "profiles 2/3 require 10- or 12-bit samples"
        );
    } else {
        assert_eq!(config.bit_depth, 8, "profiles 0/1 require 8-bit samples");
    }
    if config.profile == 0 || config.profile == 2 {
        assert_eq!(
            (config.subsampling_x, config.subsampling_y),
            (1, 1),
            "profiles 0/2 have fixed 4:2:0 subsampling"
        );
    }

    let mut w = BitWriter::new();
    w.push_bits(2, 2); // frame_marker
    w.push_bits((config.profile & 1) as u32, 1); // profile_low_bit
    w.push_bits((config.profile >> 1) as u32, 1); // profile_high_bit
    if config.profile == 3 {
        w.push_flag(false); // reserved_zero
    }
    w.push_flag(false); // show_existing_frame
    w.push_bits(0, 1); // frame_type = KEY_FRAME
    w.push_flag(true); // show_frame
    w.push_flag(false); // error_resilient_mode
    w.push_bits(0x49, 8);
    w.push_bits(0x83, 8);
    w.push_bits(0x42, 8);
    if config.profile >= 2 {
        w.push_flag(config.bit_depth == 12);
    }
    w.push_bits(0, 3); // color_space = CS_UNKNOWN
    w.push_flag(false); // color_range
    if config.profile == 1 || config.profile == 3 {
        w.push_bits(config.subsampling_x as u32, 1);
        w.push_bits(config.subsampling_y as u32, 1);
        w.push_flag(false); // reserved_zero
    }
    w.push_bits(width - 1, 16);
    w.push_bits(height - 1, 16);
    w.push_flag(false); // render_size same as frame size
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
    w.push_bits(loop_filter_level as u32, 6);
    w.push_bits(0, 3); // sharpness
    w.push_flag(loop_filter_delta_enabled);
    if loop_filter_delta_enabled {
        w.push_flag(false); // loop_filter_delta_update: deltas stay at their reset default
    }
    w.push_bits(0, 8); // base_q_idx = 0 -> lossless
    w.push_flag(false); // delta_q_y_dc coded?
    w.push_flag(false); // delta_q_uv_dc coded?
    w.push_flag(false); // delta_q_uv_ac coded?
    segmentation.write(&mut w);
    write_tile_info(&mut w, width, config.tile_cols_log2);
    w.push_bits(header_size_in_bytes as u32, 16);
    w.finish()
}

/// Writes `tile_info()` for one tile row, deriving the legal column range from the frame width
/// exactly as spec §6.2.14 does. A terminating zero is present only when the requested value is
/// below `max_log2_tile_cols`; reaching the maximum consumes no terminator bit.
fn write_tile_info(w: &mut BitWriter, width: u32, tile_cols_log2: u32) {
    const MAX_TILE_WIDTH_B64: u32 = 64;
    const MIN_TILE_WIDTH_B64: u32 = 4;

    let sb64_cols = (width + 63) >> 6;
    let mut min_log2 = 0;
    while (MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1;
    while (sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64 {
        max_log2 += 1;
    }
    max_log2 -= 1;

    assert!(
        (min_log2..=max_log2).contains(&tile_cols_log2),
        "tile_cols_log2={tile_cols_log2} is outside the legal {min_log2}..={max_log2} range \
         for width {width}"
    );
    for _ in min_log2..tile_cols_log2 {
        w.push_flag(true);
    }
    if tile_cols_log2 < max_log2 {
        w.push_flag(false);
    }
    w.push_flag(false); // tile_rows_log2 = 0
}

/// `explicit_size`: `None` inherits ref_frame_idx[0]'s slot size (`found_ref = 1`); `Some`
/// declines all three references (`found_ref = 0`) and codes the frame size explicitly --
/// for inter frames whose size must differ from their reference (the scaled-ref tests).
pub fn build_inter_header(
    ref_frame_idx: [u8; 3],
    explicit_size: Option<(u32, u32)>,
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
    match explicit_size {
        // frame_size_with_refs: inherit ref_frame_idx[0]'s slot size.
        None => w.push_flag(true),
        Some((width, height)) => {
            for _ in 0..3 {
                w.push_flag(false); // found_ref = 0: do NOT inherit the reference's size
            }
            w.push_bits(width - 1, 16);
            w.push_bits(height - 1, 16);
        }
    }
    w.push_flag(false); // render_size same as frame size
    w.push_flag(false); // allow_high_precision_mv
    w.push_flag(false); // interpolation_filter: not switchable
    w.push_bits(0, 2); // -> EIGHTTAP_SMOOTH (LITERAL_TO_TYPE[0]); never read per-block
    w.push_flag(false); // refresh_frame_context
    w.push_flag(true); // frame_parallel_decoding_mode
    w.push_bits(0, 2); // frame_context_idx
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
/// without disturbing the others. Front half mirrors the `intra_only` branch of
/// `src/header.rs::parse_uncompressed_header`; the tail from `refresh_frame_context` onward
/// is byte-for-bit identical to `build_keyframe_header`'s (both key frames and intra_only frames
/// have `frame_is_intra = true` and share the same shared-tail parsing code after the frame
/// size).
pub fn build_intra_only_header(
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
    // profile == 0 -> color_config is not read (defaults used); see the `intra_only`
    // branch of `parse_uncompressed_header`.
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

pub fn build_keyframe_compressed_header() -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    enc.write_bool(false, 128); // read_coef_probs: txSz=TX_4X4 (only iteration), update_probs=0
    write_no_update(&mut enc, 3); // read_skip_prob
    enc.finish()
}

/// Mirrors `parse_compressed_header`'s read order (`src/compressed_header.rs`)
/// exactly. Every table update is declined (`write_no_update`), so this frame's
/// `CompressedHeaderProbs` stay at `CompressedHeaderProbs::default()` -- which is also what
/// `encode_inter_tile_forced` assumes when it needs a probability value at all (partition).
pub fn build_inter_compressed_header() -> Vec<u8> {
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
pub struct KeyBlock {
    segment_id: Option<u8>,
    skip: bool,
    y_mode: u8,
    uv_mode: u8,
}

/// Every block in this file's scenarios uses `skip = true` (no residual tokens -- the whole
/// design premise) and `uv_mode = DC_PRED`; only the segment id and y_mode vary.
pub fn kb(segment_id: Option<u8>, y_mode: u8) -> KeyBlock {
    KeyBlock {
        segment_id,
        skip: true,
        y_mode,
        uv_mode: DC_PRED,
    }
}

pub fn encode_keyframe_tile(blocks: [KeyBlock; 4], seg_tree_probs: [u8; 7]) -> Vec<u8> {
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
pub fn encode_inter_tile_forced(segment_id_per_block: [u8; 4], seg_tree_probs: [u8; 7]) -> Vec<u8> {
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
pub fn header_size(compressed: &[u8]) -> u16 {
    u16::try_from(compressed.len()).expect("compressed header exceeds the 16-bit size field")
}

pub fn assemble_frame(header: Vec<u8>, compressed: Vec<u8>, tile: Vec<u8>) -> Vec<u8> {
    let mut frame = header;
    frame.extend_from_slice(&compressed);
    frame.extend_from_slice(&tile);
    frame
}
