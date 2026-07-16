//! Deblocking (loop) filter (spec §8.8 "Loop filter process").
//!
//! Faithfully to spec §8.8, the entire frame is traversed with the following
//! nested loops (taken directly from the frame-wide traversal pseudocode at
//! the top of the section):
//!
//! ```text
//! for ( row = 0; row < MiRows; row += 8 )
//!   for ( col = 0; col < MiCols; col += 8 )
//!     for ( plane = 0; plane < 3; plane++ )
//!       for ( pass = 0; pass < 2; pass++ )
//!         superblock loop filter process( plane, pass, row, col )
//! ```
//!
//! `pass == 0` means vertical edges (left/right block boundaries), and
//! `pass == 1` means horizontal edges (top/bottom block boundaries). Since
//! the same sample can be filtered multiple times, this traversal order
//! (vertical -> horizontal, superblock raster order) must be strictly
//! followed (see the NOTE in spec §8.8).
//!
//! # Known simplifications
//!
//! - `isIntra` (`RefFrames[row][col][0] <= INTRA_FRAME`) and `modeType`
//!   (whether `YModes` is `NEARESTMV`/`NEARMV`/`NEWMV`) are read from the
//!   real `ref_frame`/`y_mode` values added to `MiInfo` in M3 (see
//!   `superblock_loop_filter`).
//! - `loop_filter_ref_deltas`/`loop_filter_mode_deltas` persist across frames
//!   as spec §7.2 requires: `Decoder` (`lib.rs`) stores the previous frame's
//!   deltas and passes them into `parse_loop_filter_params`, which seeds from
//!   them (or from the default values `[1, 0, -1, -1]`/`[0, 0]` when
//!   `FrameIsIntra || error_resilient_mode`, i.e. `setup_past_independence`).

use crate::common::{get_uv_tx_size, INTRA_FRAME, MAX_SEGMENTS};
use crate::framebuffer::Plane;
use crate::header::{LoopFilterParams, SegmentationParams, SEG_LVL_ALT_L};
use crate::prob_tables::{
    BLOCK_16X16, NEARESTMV, NEARMV, NEWMV, NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP,
    TX_16X16, TX_4X4, TX_8X8,
};
use crate::tile::MiGrid;

const MAX_REF_FRAMES: usize = 4;
const MAX_MODE_LF_DELTAS: usize = 2;
const MAX_LOOP_FILTER: i32 = 63;

/// `LvlLookup[ segmentId ][ ref ][ mode ]` (spec §8.8.1).
type LvlLookup = [[[i32; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]; MAX_SEGMENTS];

#[inline]
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

#[inline]
fn clip3(low: i32, high: i32, v: i32) -> i32 {
    v.clamp(low, high)
}

/// Spec §8.8.1 "Loop filter frame init process".
fn build_lvl_lookup(lf: &LoopFilterParams, seg: &SegmentationParams) -> LvlLookup {
    // nShift is computed once from the frame-level loop_filter_level
    // (note: not from lvlSeg — this matches the spec text).
    let n_shift = (lf.level as i32) >> 5;
    let mut table: LvlLookup = [[[0; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]; MAX_SEGMENTS];

    for (segment_id, seg_table) in table.iter_mut().enumerate() {
        // Step 1-2 (spec §8.8.1): lvlSeg starts at loop_filter_level, then
        // seg_feature_active(SEG_LVL_ALT_L) overrides it (absolute or delta).
        let mut lvl_seg = lf.level as i32;
        if seg.enabled && seg.feature_enabled[segment_id][SEG_LVL_ALT_L] {
            let data = seg.feature_data[segment_id][SEG_LVL_ALT_L];
            lvl_seg = if seg.abs_or_delta_update {
                data
            } else {
                data + lf.level as i32
            };
            lvl_seg = clip3(0, MAX_LOOP_FILTER, lvl_seg);
        }

        if !lf.delta_enabled {
            for r in seg_table.iter_mut() {
                for m in r.iter_mut() {
                    *m = lvl_seg;
                }
            }
        } else {
            let intra_lvl =
                lvl_seg + (lf.ref_deltas[INTRA_FRAME as usize] as i32) * (1 << n_shift);
            seg_table[INTRA_FRAME as usize][0] = clip3(0, MAX_LOOP_FILTER, intra_lvl);
            // seg_table[INTRA_FRAME][1] is not defined by the spec (the INTRA_FRAME
            // row only has mode=0). On keyframes isIntra is always true and
            // modeType is always 0, so it's never referenced.
            for (r, ref_delta) in lf.ref_deltas.iter().enumerate().skip(1) {
                for (m, mode_delta) in lf.mode_deltas.iter().enumerate() {
                    let inter_lvl = lvl_seg
                        + (*ref_delta as i32) * (1 << n_shift)
                        + (*mode_delta as i32) * (1 << n_shift);
                    seg_table[r][m] = clip3(0, MAX_LOOP_FILTER, inter_lvl);
                }
            }
        }
    }
    table
}

/// The `limit`/`blimit`/`thresh` computation from spec §8.8.4 "Adaptive
/// filter strength process". (`lvl` is looked up from `LvlLookup` by the
/// caller and passed in.)
fn adaptive_filter_strength(lvl: i32, sharpness: u8) -> (i32, i32, i32) {
    let shift = if sharpness > 4 {
        2
    } else if sharpness > 0 {
        1
    } else {
        0
    };
    let limit = if sharpness > 0 {
        clip3(1, 9 - sharpness as i32, lvl >> shift)
    } else {
        (lvl >> shift).max(1)
    };
    let blimit = 2 * (lvl + 2) + limit;
    let thresh = lvl >> 4;
    (limit, blimit, thresh)
}

/// Spec §8.8.3 "Filter size process".
#[allow(clippy::too_many_arguments)]
fn filter_size_process(
    tx_sz: u8,
    is_32_edge: bool,
    pass: u32,
    x: u32,
    y: u32,
    sub_x: u32,
    sub_y: u32,
    mi_cols: u32,
    mi_rows: u32,
) -> u8 {
    let base_size = if tx_sz == TX_4X4 && is_32_edge {
        TX_8X8
    } else {
        tx_sz.min(TX_16X16)
    };

    let luma_boundary = (pass == 0 && sub_x == 1 && (x >> 3) == mi_cols - 1)
        || (pass == 1 && sub_y == 1 && (y >> 3) == mi_rows - 1);
    if base_size == TX_16X16 && luma_boundary {
        TX_8X8
    } else {
        base_size
    }
}

/// Reads the sample at position `x + dx*k, y + dy*k` (the general form of the
/// spec's `q_k`/`p_k`: negative `k` denotes the `p` side, non-negative `k`
/// denotes the `q` side).
#[inline]
fn get_off(plane: &Plane, x: usize, y: usize, dx: i64, dy: i64, k: i64) -> i32 {
    let px = (x as i64 + dx * k) as usize;
    let py = (y as i64 + dy * k) as usize;
    plane.get(px, py) as i32
}

#[inline]
fn set_off(plane: &mut Plane, x: usize, y: usize, dx: i64, dy: i64, k: i64, v: i32) {
    debug_assert!(
        (0..=255).contains(&v),
        "loop filter output out of 8bit range: {v}"
    );
    let px = (x as i64 + dx * k) as usize;
    let py = (y as i64 + dy * k) as usize;
    plane.set(px, py, v as u8);
}

/// Spec §8.8.5.1 "Filter mask process". Returns `(hevMask, filterMask, flatMask, flatMask2)`.
/// Since `BitDepth == 8` is fixed (this decoder only supports 8bit, see
/// `DecodeError::UnsupportedBitDepth`), the spec's bit-depth scaling via `<< (BitDepth - 8)`
/// is omitted as an identity operation (shift amount 0).
#[allow(clippy::too_many_arguments)]
fn compute_filter_mask(
    plane: &Plane,
    x: usize,
    y: usize,
    dx: i64,
    dy: i64,
    limit: i32,
    blimit: i32,
    thresh: i32,
    filter_size: u8,
) -> (bool, bool, bool, bool) {
    let g = |k: i64| get_off(plane, x, y, dx, dy, k);
    let q0 = g(0);
    let q1 = g(1);
    let q2 = g(2);
    let q3 = g(3);
    let p0 = g(-1);
    let p1 = g(-2);
    let p2 = g(-3);
    let p3 = g(-4);

    let hev_mask = (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh;

    let mut mask = (p3 - p2).abs() > limit;
    mask |= (p2 - p1).abs() > limit;
    mask |= (p1 - p0).abs() > limit;
    mask |= (q1 - q0).abs() > limit;
    mask |= (q2 - q1).abs() > limit;
    mask |= (q3 - q2).abs() > limit;
    mask |= (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit;
    let filter_mask = !mask;

    let mut flat_mask = false;
    if filter_size >= TX_8X8 {
        const THRESHOLD: i32 = 1;
        let mut m = (p1 - p0).abs() > THRESHOLD;
        m |= (q1 - q0).abs() > THRESHOLD;
        m |= (p2 - p0).abs() > THRESHOLD;
        m |= (q2 - q0).abs() > THRESHOLD;
        m |= (p3 - p0).abs() > THRESHOLD;
        m |= (q3 - q0).abs() > THRESHOLD;
        flat_mask = !m;
    }

    let mut flat_mask2 = false;
    if filter_size >= TX_16X16 {
        let q4 = g(4);
        let q5 = g(5);
        let q6 = g(6);
        let q7 = g(7);
        let p4 = g(-5);
        let p5 = g(-6);
        let p6 = g(-7);
        let p7 = g(-8);
        const THRESHOLD: i32 = 1;
        let mut m = (p7 - p0).abs() > THRESHOLD;
        m |= (q7 - q0).abs() > THRESHOLD;
        m |= (p6 - p0).abs() > THRESHOLD;
        m |= (q6 - q0).abs() > THRESHOLD;
        m |= (p5 - p0).abs() > THRESHOLD;
        m |= (q5 - q0).abs() > THRESHOLD;
        m |= (p4 - p0).abs() > THRESHOLD;
        m |= (q4 - q0).abs() > THRESHOLD;
        flat_mask2 = !m;
    }

    (hev_mask, filter_mask, flat_mask, flat_mask2)
}

/// Spec §8.8.5.2 "Narrow filter process" (`filter4`). `BitDepth == 8` fixed.
fn narrow_filter(plane: &mut Plane, x: usize, y: usize, dx: i64, dy: i64, hev_mask: bool) {
    let clamp4 = |v: i32| clip3(-128, 127, v);

    let q0 = get_off(plane, x, y, dx, dy, 0);
    let q1 = get_off(plane, x, y, dx, dy, 1);
    let p0 = get_off(plane, x, y, dx, dy, -1);
    let p1 = get_off(plane, x, y, dx, dy, -2);

    let ps1 = p1 - 0x80;
    let ps0 = p0 - 0x80;
    let qs0 = q0 - 0x80;
    let qs1 = q1 - 0x80;

    let mut filter = if hev_mask { clamp4(ps1 - qs1) } else { 0 };
    filter = clamp4(filter + 3 * (qs0 - ps0));
    let filter1 = clamp4(filter + 4) >> 3;
    let filter2 = clamp4(filter + 3) >> 3;

    let oq0 = clamp4(qs0 - filter1) + 0x80;
    let op0 = clamp4(ps0 + filter2) + 0x80;
    set_off(plane, x, y, dx, dy, 0, oq0);
    set_off(plane, x, y, dx, dy, -1, op0);

    if !hev_mask {
        let filter = round2(filter1, 1);
        let oq1 = clamp4(qs1 - filter) + 0x80;
        let op1 = clamp4(ps1 + filter) + 0x80;
        set_off(plane, x, y, dx, dy, 1, oq1);
        set_off(plane, x, y, dx, dy, -2, op1);
    }
}

/// Spec §8.8.5.3 "Wide filter process". `log2_size` is 3 (8-tap) or 4 (16-tap).
fn wide_filter(plane: &mut Plane, x: usize, y: usize, dx: i64, dy: i64, log2_size: u32) {
    let n: i64 = (1i64 << (log2_size - 1)) - 1;
    // F's index is -n..n-1. Stored in a 0-based array by adding offset n.
    let mut f = [0i32; 16];

    let mut i = -n;
    while i < n {
        let mut t = get_off(plane, x, y, dx, dy, i);
        let mut j = -n;
        while j <= n {
            let p = clip3(-((n + 1) as i32), n as i32, (i + j) as i32) as i64;
            t += get_off(plane, x, y, dx, dy, p);
            j += 1;
        }
        f[(i + n) as usize] = round2(t, log2_size);
        i += 1;
    }

    let mut i = -n;
    while i < n {
        set_off(plane, x, y, dx, dy, i, f[(i + n) as usize]);
        i += 1;
    }
}

/// Spec §8.8.5 "Sample filtering process".
#[allow(clippy::too_many_arguments)]
fn sample_filtering(
    plane: &mut Plane,
    x: usize,
    y: usize,
    dx: i64,
    dy: i64,
    limit: i32,
    blimit: i32,
    thresh: i32,
    filter_size: u8,
) {
    let (hev_mask, filter_mask, flat_mask, flat_mask2) =
        compute_filter_mask(plane, x, y, dx, dy, limit, blimit, thresh, filter_size);

    if !filter_mask {
        return;
    }
    if filter_size == TX_4X4 || !flat_mask {
        narrow_filter(plane, x, y, dx, dy, hev_mask);
    } else if filter_size == TX_8X8 || !flat_mask2 {
        wide_filter(plane, x, y, dx, dy, 3);
    } else {
        wide_filter(plane, x, y, dx, dy, 4);
    }
}

/// Spec §8.8.2 "Superblock loop filter process".
#[allow(clippy::too_many_arguments)]
fn superblock_loop_filter(
    planes: &mut [Plane; 3],
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lvl_lookup: &LvlLookup,
    sharpness: u8,
    plane_idx: usize,
    pass: u32,
    row: u32,
    col: u32,
) {
    let (sub_x, sub_y) = if plane_idx == 0 {
        (0u32, 0u32)
    } else {
        (subsampling_x, subsampling_y)
    };

    let (dx, dy, sub, edge_len): (i64, i64, u32, u32) = if pass == 0 {
        (1, 0, sub_x, 64 >> sub_y)
    } else {
        (0, 1, sub_y, 64 >> sub_x)
    };

    let num_edges = 16u32 >> sub;
    for edge in 0..num_edges {
        for i in 0..edge_len {
            let (x, y): (u32, u32) = if pass == 0 {
                (col * 8 + edge * (4 << sub_x), row * 8 + (i << sub_y))
            } else {
                (col * 8 + (i << sub_x), row * 8 + edge * (4 << sub_y))
            };

            let loop_col = ((x >> 3) >> sub_x) << sub_x;
            let loop_row = ((y >> 3) >> sub_y) << sub_y;
            let mi = mi_grid.get(loop_row, loop_col);

            let mi_size = mi.mi_size;
            let tx_size = mi.tx_size;
            let tx_sz = if plane_idx > 0 {
                get_uv_tx_size(mi_size, tx_size, subsampling_x, subsampling_y)
            } else {
                tx_size
            };
            let sb_size = if sub == 0 {
                mi_size
            } else {
                mi_size.max(BLOCK_16X16)
            };
            let skip = mi.skip;
            // Spec §8.8.2 step 9: isIntra = RefFrames[loopRow][loopCol][0] <= INTRA_FRAME.
            let ref_frame = mi.ref_frame[0];
            let is_intra = ref_frame == INTRA_FRAME;

            let is_block_edge = if pass == 0 {
                x % (8 * NUM_8X8_BLOCKS_WIDE_LOOKUP[sb_size as usize] as u32) == 0
            } else {
                y % (8 * NUM_8X8_BLOCKS_HIGH_LOOKUP[sb_size as usize] as u32) == 0
            };

            let is_tx_edge = if pass == 1
                && sub_x == 1
                && mi_cols % 2 == 1
                && edge % 2 == 1
                && (x + 8) >= mi_cols * 8
            {
                false
            } else {
                edge % (1 << tx_sz) == 0
            };

            let is_32_edge = edge % 8 == 0;

            let on_screen = !(x >= 8 * mi_cols
                || y >= 8 * mi_rows
                || (pass == 0 && x == 0)
                || (pass == 1 && y == 0));

            let apply_filter = on_screen && (is_block_edge || (is_tx_edge && (is_intra || !skip)));

            let filter_size = filter_size_process(
                tx_sz, is_32_edge, pass, x, y, sub_x, sub_y, mi_cols, mi_rows,
            );

            // Spec §8.8.4: modeType = 1 if mode in {NEARESTMV,NEARMV,NEWMV} else 0.
            let mode_type = matches!(mi.y_mode, NEARESTMV | NEARMV | NEWMV) as usize;
            let lvl = lvl_lookup[mi.segment_id as usize][ref_frame as usize][mode_type];

            if apply_filter && lvl > 0 {
                let (limit, blimit, thresh) = adaptive_filter_strength(lvl, sharpness);
                sample_filtering(
                    &mut planes[plane_idx],
                    (x >> sub_x) as usize,
                    (y >> sub_y) as usize,
                    dx,
                    dy,
                    limit,
                    blimit,
                    thresh,
                    filter_size,
                );
            }
        }
    }
}

/// Frame-wide entry point for spec §8.8 "Loop filter process".
///
/// `planes` is `CurrFrame` (equivalent to `TileDecoder::planes`/`planes_mut`,
/// a buffer allocated out to superblock boundaries); `mi_grid` is the mode
/// info grid; `mi_cols`/`mi_rows` is the non-padded, mi-unit frame size
/// obtained from `compute_image_size()`.
#[allow(clippy::too_many_arguments)]
pub fn loop_filter_frame(
    planes: &mut [Plane; 3],
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
) {
    let lvl_lookup = build_lvl_lookup(lf, seg);

    let mut row = 0u32;
    while row < mi_rows {
        let mut col = 0u32;
        while col < mi_cols {
            for plane_idx in 0..3usize {
                for pass in 0..2u32 {
                    superblock_loop_filter(
                        planes,
                        mi_grid,
                        mi_cols,
                        mi_rows,
                        subsampling_x,
                        subsampling_y,
                        &lvl_lookup,
                        lf.sharpness,
                        plane_idx,
                        pass,
                        row,
                        col,
                    );
                }
            }
            col += 8;
        }
        row += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round2_matches_spec_formula() {
        assert_eq!(round2(0, 1), 0);
        assert_eq!(round2(1, 1), 1);
        assert_eq!(round2(3, 3), 0); // (3+4)>>3 = 0
        assert_eq!(round2(4, 3), 1); // (4+4)>>3 = 1
    }

    /// A `SegmentationParams` with segmentation disabled (the M2 default).
    fn no_segmentation() -> SegmentationParams {
        SegmentationParams {
            enabled: false,
            update_map: false,
            tree_probs: [255; 7],
            pred_prob: [255; 3],
            temporal_update: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; 4]; MAX_SEGMENTS],
            feature_data: [[0; 4]; MAX_SEGMENTS],
        }
    }

    #[test]
    fn lvl_lookup_without_deltas_is_flat_level() {
        let lf = LoopFilterParams {
            level: 20,
            sharpness: 0,
            delta_enabled: false,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        };
        let table = build_lvl_lookup(&lf, &no_segmentation());
        for seg in table.iter() {
            for r in seg.iter() {
                for m in r.iter() {
                    assert_eq!(*m, 20);
                }
            }
        }
    }

    #[test]
    fn lvl_lookup_applies_intra_ref_delta() {
        // level=40 -> nShift = 40>>5 = 1. Default of ref_deltas[INTRA_FRAME] is 1.
        let lf = LoopFilterParams {
            level: 40,
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        };
        let table = build_lvl_lookup(&lf, &no_segmentation());
        // intraLvl = 40 + 1*(1<<1) = 42
        assert_eq!(table[0][INTRA_FRAME as usize][0], 42);
    }

    #[test]
    fn lvl_lookup_seg_lvl_alt_l_absolute_override() {
        let lf = LoopFilterParams {
            level: 20,
            sharpness: 0,
            delta_enabled: false,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        };
        let mut seg = no_segmentation();
        seg.enabled = true;
        seg.abs_or_delta_update = true;
        seg.feature_enabled[3][SEG_LVL_ALT_L] = true;
        seg.feature_data[3][SEG_LVL_ALT_L] = 50;
        let table = build_lvl_lookup(&lf, &seg);
        // Segment 3 uses the absolute override (50); other segments stay flat at 20.
        for m in table[3].iter().flatten() {
            assert_eq!(*m, 50);
        }
        for m in table[0].iter().flatten() {
            assert_eq!(*m, 20);
        }
    }

    #[test]
    fn lvl_lookup_seg_lvl_alt_l_delta_is_clipped() {
        let lf = LoopFilterParams {
            level: 60,
            sharpness: 0,
            delta_enabled: false,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        };
        let mut seg = no_segmentation();
        seg.enabled = true;
        seg.abs_or_delta_update = false;
        seg.feature_enabled[2][SEG_LVL_ALT_L] = true;
        seg.feature_data[2][SEG_LVL_ALT_L] = 10; // 60 + 10 = 70, clipped to 63.
        let table = build_lvl_lookup(&lf, &seg);
        for m in table[2].iter().flatten() {
            assert_eq!(*m, 63);
        }
    }

    #[test]
    fn adaptive_filter_strength_zero_sharpness() {
        let (limit, blimit, thresh) = adaptive_filter_strength(20, 0);
        assert_eq!(limit, 20);
        assert_eq!(blimit, 2 * (20 + 2) + 20);
        assert_eq!(thresh, 20 >> 4);
    }

    #[test]
    fn narrow_filter_flat_input_is_noop_like() {
        // A perfectly flat input (all the same value) should be unchanged by filtering.
        let mut p = Plane::new(8, 1);
        for x in 0..8 {
            p.set(x, 0, 128);
        }
        narrow_filter(&mut p, 4, 0, 1, 0, false);
        for x in 0..8 {
            assert_eq!(p.get(x, 0), 128);
        }
    }
}
