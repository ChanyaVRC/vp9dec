//! Synthetic coverage for the reference-frame-scaling bounds (spec §8.5.2.3).
//!
//! The spec's conformance requirement `2 * FrameWidth >= RefFrameWidth[ refIdx ]` (and the
//! height analog) caps the motion-compensation scaling step at 32 (1/16-pel units), and every
//! scaled inter-prediction scratch buffer -- scalar and AVX2 (`MAX_INTERMEDIATE_HEIGHT` in
//! `src/predict.rs`) -- is sized to exactly that bound. A malformed stream is free to violate
//! it (e.g. a 64x256 keyframe followed by a 64x64 inter frame referencing it, a 4x vertical
//! ratio), so the decoder must reject any block using the out-of-range reference upstream of
//! prediction (`TileError::RefFrameSizeOutOfRange`, `decode_block`'s malformed-bitstream
//! guards in `src/tile.rs`), not panic on the scratch bounds. The check is per-block, not at
//! reference resolution: a conformant stream may *list* an out-of-range slot it never
//! predicts from (`vp90-2-22-svc_1280x720_3`'s base layer lists a 4x-larger enhancement
//! frame), so eager rejection would break the official sweep.
//!
//! Two synthetic streams, hand-built with `tests/common/encoder.rs`'s builders (the same
//! machinery as `synthetic_seg_test.rs`, passing explicit frame sizes instead of that
//! module's default 16x16):
//!
//! 1. ratio 4.0 (64x256 keyframe -> 64x64 inter): must return `Err`, never panic -- before the
//!    upstream rejection existed, this stream panicked on the scalar scratch's slice bounds
//!    (`y_step = 64 > 32`, `intermediate_height = 260 > 134`).
//! 2. ratio exactly 2.0 (64x128 keyframe -> 64x64 inter): the spec's boundary case
//!    (`y_step = 32`, `intermediate_height = 134` -- exactly the scratch size) must DECODE,
//!    proving the rejection threshold is not tighter than the conformance bound. The all-gray
//!    oracle also pins the scaled prediction output (a constant plane is FIR-invariant).
//!
//! The inter frame forces `SEG_LVL_SKIP` + `SEG_LVL_REF_FRAME = LAST` on segment 0 (with
//! `segmentation_update_map = 0`, so every block predicts segment 0 from the all-zero previous
//! map): per spec §6.4.16 the block then reads no skip/is_inter/ref/mode/MV bits at all --
//! `y_mode` is forced to `ZEROMV` -- so the tile data is just one partition symbol, and the
//! block still runs full scaled motion compensation (ZEROMV does not bypass ref scaling).

mod common;

use common::encoder::{
    assemble_frame, build_inter_compressed_header, build_inter_header,
    build_keyframe_compressed_header, build_keyframe_header, encode_tree, header_size, SegSpec,
};
use vp9dec::header::{SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use vp9dec::prob_tables::{
    DC_PRED, DEFAULT_PARTITION_PROBS, DEFAULT_SKIP_PROB, INTRA_MODE_TREE, KF_PARTITION_PROBS,
    KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, LAST_FRAME, PARTITION_NONE, PARTITION_TREE,
};
use vp9dec::test_support::BoolEncoder;
use vp9dec::tile::TileError;
use vp9dec::{DecodeError, Decoder, PlaneData};

/// All frames here are one superblock (64) wide, so `Sb64Cols == 1`: `tile_info()`'s
/// tile-cols loop reads no bits and `tile_rows_log2` is a single 0 bit -- the tiling tail
/// `tests/common/encoder.rs`'s builders write.
const WIDTH: u32 = 64;

/// Tile data for a 64-wide keyframe of `sb_rows` stacked 64x64 superblocks, each a single
/// `PARTITION_NONE` DC_PRED block with `skip = 1` (lossless -> `read_tx_size` reads no bits).
/// Contexts, hand-tracked: the partition ctx is 12 for every SB (`bsl = 3`; a 64x64 NONE
/// writes `15 >> 4 == 0` into both partition context arrays, so the above/left bits never
/// set), and the skip ctx is 0 for the first SB (no neighbors) then 1 (above MI has skip = 1,
/// left is unavailable at column 0). Above/left y modes are all DC_PRED.
fn encode_dc_keyframe_tile(sb_rows: usize) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    for sb in 0..sb_rows {
        encode_tree(&mut enc, &PARTITION_TREE, PARTITION_NONE, |n| {
            KF_PARTITION_PROBS[12][n]
        });
        let skip_ctx = if sb == 0 { 0 } else { 1 };
        enc.write_bool(true, DEFAULT_SKIP_PROB[skip_ctx]);
        encode_tree(&mut enc, &INTRA_MODE_TREE, DC_PRED, |n| {
            KF_Y_MODE_PROBS[DC_PRED as usize][DC_PRED as usize][n]
        });
        encode_tree(&mut enc, &INTRA_MODE_TREE, DC_PRED, |n| {
            KF_UV_MODE_PROBS[DC_PRED as usize][n]
        });
    }
    enc.finish()
}

/// Segmentation forcing `SEG_LVL_SKIP` + `SEG_LVL_REF_FRAME = LAST` on segment 0, with the
/// segment map *predicted* (`update_map = 0` -> all blocks take segment 0 from the cleared
/// previous map, no per-block seg-id bits).
fn forced_skip_last_seg() -> SegSpec {
    let mut seg = SegSpec {
        enabled: true,
        update_map: false,
        update_data: true,
        ..SegSpec::disabled()
    };
    seg.feature_enabled[0][SEG_LVL_REF_FRAME] = true;
    seg.feature_data[0][SEG_LVL_REF_FRAME] = LAST_FRAME as i32;
    seg.feature_enabled[0][SEG_LVL_SKIP] = true;
    seg
}

/// One 64x64 inter frame: a single `PARTITION_NONE` superblock (ctx 12, as in the keyframe,
/// but at the inter table). With segment 0's forced features, that partition symbol is the
/// tile's ONLY syntax element (see the module doc).
fn encode_forced_zeromv_inter_tile() -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    encode_tree(&mut enc, &PARTITION_TREE, PARTITION_NONE, |n| {
        DEFAULT_PARTITION_PROBS[12][n]
    });
    enc.finish()
}

/// Builds the two-frame scenario: a `64 x ref_height` all-gray keyframe, then a 64x64
/// ZEROMV inter frame referencing it (vertical scaling ratio `ref_height / 64`).
fn build_scaled_ref_frames(ref_height: u32) -> (Vec<u8>, Vec<u8>) {
    let kf_compressed = build_keyframe_compressed_header();
    let kf_header = build_keyframe_header(
        WIDTH,
        ref_height,
        0,
        false,
        &SegSpec::disabled(),
        header_size(&kf_compressed),
    );
    let kf_tile = encode_dc_keyframe_tile(ref_height as usize / 64);
    let keyframe = assemble_frame(kf_header, kf_compressed, kf_tile);

    let seg = forced_skip_last_seg();
    let inter_compressed = build_inter_compressed_header();
    // ref_frame_idx all 0 (the keyframe refreshed every slot); the explicit 64x64 size is the
    // whole point -- an inter frame SMALLER than its reference.
    let inter_header = build_inter_header(
        [0, 0, 0],
        Some((64, 64)),
        0,
        &seg,
        header_size(&inter_compressed),
    );
    let inter_tile = encode_forced_zeromv_inter_tile();
    let inter = assemble_frame(inter_header, inter_compressed, inter_tile);

    (keyframe, inter)
}

fn assert_all_gray(plane: &PlaneData, what: &str) {
    match plane {
        PlaneData::U8(data) => {
            assert!(!data.is_empty(), "{what}: empty plane");
            assert!(
                data.iter().all(|&v| v == 128),
                "{what}: expected an all-128 plane"
            );
        }
        PlaneData::U16(_) => panic!("{what}: expected an 8-bit plane"),
    }
}

/// A block predicting from a reference more than 2x the current frame's height (here 4x:
/// 256 -> 64) violates spec §8.5.2.3's conformance bound; the decoder must reject it with a
/// clean `Err` before prediction -- historically this stream PANICKED on the scaled scalar
/// path's fixed scratch (`y_step = 64`, `intermediate_height = 260 >
/// MAX_INTERMEDIATE_HEIGHT = 134`).
#[test]
fn oversized_reference_scaling_ratio_is_rejected_not_panicked() {
    let (keyframe, inter) = build_scaled_ref_frames(256);
    let mut dec = Decoder::new();
    let decoded = dec
        .decode_frame(&keyframe)
        .expect("64x256 keyframe should decode");
    assert_all_gray(&decoded[0].frame.as_ref().unwrap().y, "keyframe y");

    let err = dec
        .decode_frame(&inter)
        .expect_err("a 4x vertical reference scaling ratio must be rejected");
    assert!(
        matches!(err, DecodeError::Tile(TileError::RefFrameSizeOutOfRange)),
        "expected RefFrameSizeOutOfRange, got {err:?}"
    );
}

/// Ratio exactly 2.0 is the spec's conformance boundary (`y_step = 32`,
/// `intermediate_height` exactly `MAX_INTERMEDIATE_HEIGHT`): it must decode -- the rejection
/// threshold must not be tighter than the bound the official `vp90-2-13-largescaling` /
/// resize vectors are entitled to reach -- and the scaled prediction of a constant-gray
/// reference must stay constant gray (the subpel FIR sums to unity).
#[test]
fn reference_at_exactly_twice_frame_size_decodes() {
    let (keyframe, inter) = build_scaled_ref_frames(128);
    let mut dec = Decoder::new();
    dec.decode_frame(&keyframe)
        .expect("64x128 keyframe should decode");

    let decoded = dec
        .decode_frame(&inter)
        .expect("a 2x reference scaling ratio is conformant and must decode");
    let frame = decoded[0].frame.as_ref().unwrap();
    assert_eq!((frame.width, frame.height), (64, 64));
    assert_all_gray(&frame.y, "scaled inter y");
    assert_all_gray(&frame.u, "scaled inter u");
    assert_all_gray(&frame.v, "scaled inter v");
}
