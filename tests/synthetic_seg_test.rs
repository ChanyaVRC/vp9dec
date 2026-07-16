//! Synthetic, pure-std, round-trip coverage for the three segmentation features that no
//! official (or readily-encodable) libvpx IVF vector exercises: `SEG_LVL_ALT_L`,
//! `SEG_LVL_REF_FRAME`, `SEG_LVL_SKIP` (see `docs/implementation-notes.md`, "Conformance
//! coverage" section, and its 2026-07-14 follow-up entry for the full writeup).
//!
//! This file's VP9 *encoder* (bool coder + uncompressed-header bit writer + frame/tile
//! builders) lives in `tests/common/encoder.rs` (moved there in Wave 3 test-layer
//! consolidation, 2026-07-16, so future synthetic-vector test files -- e.g. M4's
//! reference-frame-scaling coverage -- can reuse it). This file keeps the scenario builders
//! (`build_skip_ref_frames`/`build_steering_frames`/`build_alt_l_frames`) and the tests: it
//! builds just enough bitstream -- a handful of skip=1 intra blocks and one all-forced-fields
//! inter frame -- to drive the three features through `vp9dec::Decoder`, then checks the
//! decoded output.
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

mod common;

use common::encoder::{
    assemble_frame, build_inter_compressed_header, build_inter_header, build_intra_only_header,
    build_keyframe_compressed_header, build_keyframe_header, encode_inter_tile_forced,
    encode_keyframe_tile, header_size, kb, SegSpec, HEIGHT, WIDTH,
};
use vp9dec::header::{SEG_LVL_ALT_L, SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use vp9dec::prob_tables::{GOLDEN_FRAME, H_PRED, LAST_FRAME, V_PRED};
use vp9dec::{DecodedFrame, Decoder};

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
    let keyframe_header = build_keyframe_header(
        0,
        false,
        &SegSpec::disabled(),
        header_size(&keyframe_compressed),
    );
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
    let keyframe_header = build_keyframe_header(
        0,
        false,
        &SegSpec::disabled(),
        header_size(&keyframe_compressed),
    );
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
    let header = build_keyframe_header(30, false, &seg, header_size(&compressed));
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
// Test: frame-level loop_filter_level == 0 must skip the loop filter outright (spec §8.1 step
// 2), even though loop_filter_delta_enabled can still make build_lvl_lookup's per-block level
// (spec §8.8.1) nonzero via loop_filter_ref_deltas. See docs/implementation-notes.md's "M4 wave
// 2" entry for the full root-cause writeup (this pins the fix for a real category-A sweep
// failure, `vp90-2-00-quantizer-00..07`/`vp90-2-13-largescaling`).
// ===========================================================================================

/// Same V_PRED/H_PRED 127/129-edge key frame as `build_skip_ref_frames`'s keyframe (segmentation
/// disabled), but with `loop_filter_delta_enabled = true` and `loop_filter_level = 0`. A key
/// frame's `ref_deltas` are always the spec §7.2 default `[1, 0, -1, -1]`
/// (`setup_past_independence`), so `loop_filter_ref_deltas[INTRA_FRAME] == 1` here.
fn build_delta_enabled_zero_level_frame() -> Vec<u8> {
    let compressed = build_keyframe_compressed_header();
    let header = build_keyframe_header(0, true, &SegSpec::disabled(), header_size(&compressed));
    let tile = encode_keyframe_tile(
        [
            kb(None, V_PRED),
            kb(None, V_PRED),
            kb(None, H_PRED),
            kb(None, H_PRED),
        ],
        [128; 7],
    );
    assemble_frame(header, compressed, tile)
}

#[test]
fn loop_filter_level_zero_skips_filtering_despite_nonzero_ref_delta() {
    let mut decoder = Decoder::new();
    let frame_bytes = build_delta_enabled_zero_level_frame();
    let frame = decode_single(
        &mut decoder,
        &frame_bytes,
        "delta-enabled, level-0 key frame",
    )
    .frame
    .expect("key frame has show_frame = 1");

    // Same edge and narrow_filter math as the ALT_L test above: with lvlSeg=0 and nShift=0,
    // `build_lvl_lookup` computes intraLvl = 0 + (ref_deltas[INTRA_FRAME]=1 << 0) = 1, a
    // nonzero per-block level. If `Decoder` still invoked the loop filter on that level (the
    // bug this test pins), the identical hev_mask=false/filter=6/filter1=filter2=1 derivation
    // from the ALT_L test would flatten rows 6-9 to 128 regardless of the level being only 1
    // instead of 63 (the narrow filter's output magnitude doesn't scale with lvl once the
    // filter_mask/flat_mask gates pass) -- so seeing 127/129 here can only mean the whole
    // process was skipped per spec §8.1 step 2, not that it ran and produced a no-op delta.
    for x in 0..WIDTH as usize {
        assert_eq!(
            frame.y[7 * WIDTH as usize + x],
            127,
            "row7: must stay unfiltered"
        );
        assert_eq!(
            frame.y[8 * WIDTH as usize + x],
            129,
            "row8: must stay unfiltered"
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
