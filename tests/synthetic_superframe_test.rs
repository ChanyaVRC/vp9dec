//! Synthetic coverage for the M4 wave 3 fix: within one container chunk (superframe), only the
//! LAST constituent frame's pixels may surface as `DecodedFrame::frame`, even when an earlier
//! constituent's own `show_frame` is also 1 (spec §5.26 permits multiple output frames per
//! superframe, but this crate's conformance oracle -- libvpx, via ffmpeg -- does not produce
//! them; see `docs/implementation-notes.md`, "M4 wave 3").
//!
//! Reuses `tests/common/encoder.rs`'s keyframe builder (both constituents are independent key
//! frames -- `build_keyframe_header` always signals `show_frame = 1` -- so no inter/reference
//! plumbing is needed to construct the scenario) plus a hand-built superframe index (mirroring
//! `src/superframe.rs`'s own test helper) to pack them into one chunk.

mod common;

use common::encoder::{
    assemble_frame, build_keyframe_compressed_header, build_keyframe_header, encode_keyframe_tile,
    header_size, kb, SegSpec, HEIGHT, WIDTH,
};
use vp9dec::prob_tables::{DC_PRED, H_PRED, V_PRED};
use vp9dec::Decoder;

/// One self-contained key frame (uncompressed header + compressed header + tile data),
/// `WIDTH x HEIGHT` (16x16, a 2x2 grid of 8x8 blocks), with `y_modes[i]` as block `i`'s
/// (`kb`'s) intra mode. Segmentation and loop filtering are both off.
fn build_keyframe_frame(y_modes: [u8; 4]) -> Vec<u8> {
    let compressed = build_keyframe_compressed_header();
    let header = build_keyframe_header(0, false, &SegSpec::disabled(), header_size(&compressed));
    let tile = encode_keyframe_tile(
        [
            kb(None, y_modes[0]),
            kb(None, y_modes[1]),
            kb(None, y_modes[2]),
            kb(None, y_modes[3]),
        ],
        [128; 7],
    );
    assemble_frame(header, compressed, tile)
}

/// Hand-builds a superframe index for the given frame sizes, using 1-byte frame sizes (every
/// frame `build_keyframe_frame` produces here is well under 256 bytes) -- same layout as
/// `src/superframe.rs`'s own private `#[cfg(test)]` helper of the same name, duplicated here
/// since that one isn't reachable from an integration test.
fn build_index_1byte(frame_sizes: &[u8]) -> Vec<u8> {
    // frame_size_length_minus_one = 0 (1-byte sizes), so its `<< 3` field is 0 and only the
    // `num_frames_minus_one` low 3 bits vary.
    let marker = 0xc0 | (frame_sizes.len() as u8 - 1);
    let mut index = vec![marker];
    index.extend_from_slice(frame_sizes);
    index.push(marker);
    index
}

fn pack_superframe(frames: &[Vec<u8>]) -> Vec<u8> {
    let sizes: Vec<u8> = frames
        .iter()
        .map(|f| u8::try_from(f.len()).expect("test frame exceeds a 1-byte superframe size"))
        .collect();
    let mut data: Vec<u8> = frames.iter().flatten().copied().collect();
    data.extend(build_index_1byte(&sizes));
    data
}

/// Both constituents have `show_frame == 1` (true of every frame `build_keyframe_header`
/// builds); only the second's pixels must be reported as this chunk's displayed frame.
#[test]
fn superframe_with_multiple_shown_constituents_only_outputs_the_last() {
    let frame_a = build_keyframe_frame([DC_PRED; 4]);
    let frame_b = build_keyframe_frame([V_PRED, V_PRED, H_PRED, H_PRED]);
    let chunk = pack_superframe(&[frame_a, frame_b]);

    let mut decoder = Decoder::new();
    let decoded = decoder
        .decode_frame(&chunk)
        .expect("2-constituent superframe should decode");
    assert_eq!(
        decoded.len(),
        2,
        "decode stats must still be reported for both constituents"
    );

    assert!(
        decoded[0].info.is_some(),
        "the first constituent's own decode stats are still observable"
    );
    assert!(
        decoded[0].frame.is_none(),
        "the first constituent's own show_frame == 1 must not surface an output frame when \
         a later constituent follows it in the same chunk"
    );

    assert!(decoded[1].info.is_some());
    let shown = decoded[1]
        .frame
        .as_ref()
        .expect("the last constituent's frame is the chunk's displayed output");
    // Frame B's V_PRED (no "above" neighbor, row 0) / H_PRED (no "left" neighbor, row 1) split
    // -- the same 127/129 derivation `tests/synthetic_seg_test.rs` uses for its keyframes
    // (`src/predict.rs::predict_intra`: `base - 1`/`base + 1` when a neighbor is unavailable).
    // Frame A's would-be content (flat 128, from DC_PRED with no neighbors) is deliberately
    // never checked -- it must never surface at all, which the assertion above already covers.
    assert_eq!(shown.y[0], 127, "must be frame B's content, not frame A's");
    assert_eq!(shown.y[(HEIGHT as usize / 2) * WIDTH as usize], 129);
}

/// A single-frame chunk (no superframe index) is unaffected: the sole constituent's own
/// `show_frame` still determines whether it's displayed, exactly as before this wave's change.
#[test]
fn single_frame_chunk_is_unaffected() {
    let chunk = build_keyframe_frame([DC_PRED; 4]);

    let mut decoder = Decoder::new();
    let decoded = decoder
        .decode_frame(&chunk)
        .expect("single key frame should decode");
    assert_eq!(decoded.len(), 1);
    assert!(
        decoded[0].frame.is_some(),
        "the sole constituent's show_frame == 1 must still surface an output frame"
    );
}
