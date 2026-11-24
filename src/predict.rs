//! Prediction processes: intra prediction (spec §8.5.1) and inter prediction
//! (spec §8.5.2). Both are called per plane by [`crate::tile::TileDecoder`]'s
//! residual path (`src/tile/residual.rs`).
//!
//! [`predict_intra`] implements the 10 intra modes (`DC_PRED` through
//! `TM_PRED`; VP9 has no smooth-style filters, unlike VP8 or AV1). Per
//! transform block the caller passes the [`crate::framebuffer::Plane`] to
//! predict into, the block's top-left `(x, y)`,
//! `have_left`/`have_above`/`not_on_right` (availability at frame/block
//! edges), `tx_size`, the mode, and the clip bounds `max_x`/`max_y` (= the
//! spec's `maxX`/`maxY`, which differ per plane).
//!
//! [`predict_inter`] implements motion compensation: MV selection, edge
//! clamping, and reference-frame scaling (§8.5.2.1-8.5.2.3) feeding the
//! per-block subpel interpolation (§8.5.2.4, `block_inter_predict`), plus
//! compound (two-reference) averaging. `block_inter_predict` dispatches to
//! the AVX2 kernels in `src/simd/inter.rs` (unscaled and reference-scaled) when
//! available; `block_inter_predict_scalar` is the always-kept fallback and
//! the bit-exactness oracle the SIMD unit tests pin against.

use crate::common::{clip3, round2};
use crate::framebuffer::Plane;
use crate::prob_tables::{
    BLOCK_8X8, D117_PRED, D135_PRED, D153_PRED, D207_PRED, D45_PRED, D63_PRED, DC_PRED, H_PRED,
    NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP, TM_PRED, TX_4X4, V_PRED,
};
use crate::subpel::{
    INTERP_EXTEND, REF_SCALE_SHIFT, SUBPEL_BITS, SUBPEL_FILTERS, SUBPEL_MASK, SUBPEL_SHIFTS,
};
use crate::tile::Mv;

/// Spec §8.5.1's `predict_intra` process.
///
/// `x`/`y` are absolute pixel coordinates within `plane`; `max_x`/`max_y` are
/// the spec's `maxX`/`maxY` (= the maximum coordinates that may be
/// referenced within the plane).
///
/// Dispatches the high-frequency modes (`V_PRED`, `H_PRED`, `DC_PRED`, `TM_PRED`, all sizes
/// 4/8/16/32 and all bit depths 8/10/12) to the AVX2 kernels in `src/simd/intra.rs` when
/// available; falls back to [`predict_intra_scalar`] for the six directional modes,
/// `VP9DEC_NO_SIMD=1`, and non-x86_64 targets.
#[allow(clippy::too_many_arguments)]
pub fn predict_intra(
    plane: &mut Plane,
    x: usize,
    y: usize,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    tx_size: u8,
    mode: u8,
    max_x: usize,
    max_y: usize,
    bit_depth: u8,
) {
    #[cfg(target_arch = "x86_64")]
    if matches!(mode, V_PRED | H_PRED | DC_PRED | TM_PRED) && crate::simd::avx2_enabled() {
        let log2_size = (tx_size as u32) + 2;
        let size = 1usize << log2_size;
        let pw = plane.stride();
        debug_assert!(
            x >= plane.x0 && x - plane.x0 + size <= plane.width && y + size <= plane.height,
            "predict_intra block outside its plane bounds"
        );
        let local_x = x - plane.x0;
        let base = 1u16 << (bit_depth - 1);

        match mode {
            V_PRED => {
                if !have_above {
                    // SAFETY: avx2_enabled() confirmed; block bounds checked above.
                    unsafe {
                        crate::simd::intra_pred_fill_avx2(
                            plane.as_mut_slice(),
                            pw,
                            local_x,
                            y,
                            base - 1,
                            size,
                        );
                    }
                } else if x + size - 1 <= max_x {
                    let (above_rows, dst_rows) = plane.as_mut_slice().split_at_mut(y * pw);
                    let above_start = (y - 1) * pw + local_x;
                    let above = &above_rows[above_start..above_start + size];
                    // SAFETY: avx2_enabled() confirmed; `above` is length `size` from row
                    // `y - 1` and `dst_rows` starts at row `y` (stride `pw`) with `>= size`
                    // rows remaining.
                    unsafe {
                        crate::simd::intra_pred_v_avx2(dst_rows, pw, local_x, 0, above, size);
                    }
                } else {
                    let mut above_buf = [0u16; 32];
                    for (i, slot) in above_buf[..size].iter_mut().enumerate() {
                        *slot = plane.get((x + i).min(max_x), y - 1);
                    }
                    // SAFETY: avx2_enabled() confirmed; above_buf[..size] and block bounds hold.
                    unsafe {
                        crate::simd::intra_pred_v_avx2(
                            plane.as_mut_slice(),
                            pw,
                            local_x,
                            y,
                            &above_buf[..size],
                            size,
                        );
                    }
                }
            }
            H_PRED => {
                if !have_left {
                    // SAFETY: avx2_enabled() confirmed; block bounds checked above.
                    unsafe {
                        crate::simd::intra_pred_fill_avx2(
                            plane.as_mut_slice(),
                            pw,
                            local_x,
                            y,
                            base + 1,
                            size,
                        );
                    }
                } else {
                    let mut left_buf = [0u16; 32];
                    for (i, slot) in left_buf[..size].iter_mut().enumerate() {
                        *slot = plane.get(x - 1, (y + i).min(max_y));
                    }
                    // SAFETY: avx2_enabled() confirmed; left_buf[..size] and block bounds hold.
                    unsafe {
                        crate::simd::intra_pred_h_avx2(
                            plane.as_mut_slice(),
                            pw,
                            local_x,
                            y,
                            &left_buf[..size],
                            size,
                        );
                    }
                }
            }
            DC_PRED => {
                let mut left_buf = [0u16; 32];
                let left_opt = if have_left {
                    for (i, slot) in left_buf[..size].iter_mut().enumerate() {
                        *slot = plane.get(x - 1, (y + i).min(max_y));
                    }
                    Some(&left_buf[..size])
                } else {
                    None
                };

                if have_above && x + size - 1 <= max_x {
                    let (above_rows, dst_rows) = plane.as_mut_slice().split_at_mut(y * pw);
                    let above_start = (y - 1) * pw + local_x;
                    let above = &above_rows[above_start..above_start + size];
                    // SAFETY: avx2_enabled() confirmed; neighbor slices have length `size`,
                    // samples are valid VP9 samples (<= 4095), and dst_rows has `>= size` rows.
                    unsafe {
                        crate::simd::intra_pred_dc_avx2(
                            dst_rows,
                            pw,
                            local_x,
                            0,
                            Some(above),
                            left_opt,
                            log2_size,
                            bit_depth,
                        );
                    }
                } else {
                    let mut above_buf = [0u16; 32];
                    let above_opt = if have_above {
                        for (i, slot) in above_buf[..size].iter_mut().enumerate() {
                            *slot = plane.get((x + i).min(max_x), y - 1);
                        }
                        Some(&above_buf[..size])
                    } else {
                        None
                    };
                    // SAFETY: avx2_enabled() confirmed; neighbor slices have length `size`
                    // (when `Some`) and block bounds hold.
                    unsafe {
                        crate::simd::intra_pred_dc_avx2(
                            plane.as_mut_slice(),
                            pw,
                            local_x,
                            y,
                            above_opt,
                            left_opt,
                            log2_size,
                            bit_depth,
                        );
                    }
                }
            }
            TM_PRED => {
                let top_left = if have_above && have_left {
                    let sx = x.saturating_sub(1).min(max_x);
                    plane.get(sx, y - 1)
                } else if have_above {
                    base + 1
                } else {
                    base - 1
                };
                let mut left_buf = [base + 1; 32];
                if have_left {
                    for (i, slot) in left_buf[..size].iter_mut().enumerate() {
                        *slot = plane.get(x - 1, (y + i).min(max_y));
                    }
                }

                if have_above && x + size - 1 <= max_x {
                    let (above_rows, dst_rows) = plane.as_mut_slice().split_at_mut(y * pw);
                    let above_start = (y - 1) * pw + local_x;
                    let above = &above_rows[above_start..above_start + size];
                    // SAFETY: avx2_enabled() confirmed; neighbor slices have length `size`,
                    // samples are <= 4095, and dst_rows has `>= size` rows.
                    unsafe {
                        crate::simd::intra_pred_tm_avx2(
                            dst_rows,
                            pw,
                            local_x,
                            0,
                            above,
                            &left_buf[..size],
                            top_left,
                            size,
                            bit_depth,
                        );
                    }
                } else {
                    let mut above_buf = [base - 1; 32];
                    if have_above {
                        for (i, slot) in above_buf[..size].iter_mut().enumerate() {
                            *slot = plane.get((x + i).min(max_x), y - 1);
                        }
                    }
                    // SAFETY: avx2_enabled() confirmed; neighbor buffers have length `>= size`
                    // and block bounds hold.
                    unsafe {
                        crate::simd::intra_pred_tm_avx2(
                            plane.as_mut_slice(),
                            pw,
                            local_x,
                            y,
                            &above_buf[..size],
                            &left_buf[..size],
                            top_left,
                            size,
                            bit_depth,
                        );
                    }
                }
            }
            _ => unreachable!(),
        }
        return;
    }

    predict_intra_scalar(
        plane,
        x,
        y,
        have_left,
        have_above,
        not_on_right,
        tx_size,
        mode,
        max_x,
        max_y,
        bit_depth,
    );
}

/// Scalar body of [`predict_intra`] (spec §8.5.1 verbatim): the always-kept fallback for
/// directional modes, `VP9DEC_NO_SIMD=1`, and non-x86_64, and the bit-exactness oracle
/// `tests/unit/simd.rs` pins the intra AVX2 kernels against (hence `pub(crate)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_intra_scalar(
    plane: &mut Plane,
    x: usize,
    y: usize,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    tx_size: u8,
    mode: u8,
    max_x: usize,
    max_y: usize,
    bit_depth: u8,
) {
    let log2_size = (tx_size as u32) + 2;
    let size = 1usize << log2_size;
    let base = 1i32 << (bit_depth - 1);

    // Fixed-size scratch (size <= 32, the 32x32 max transform): avoids per-block heap
    // allocations. above_row holds the spec's indices -1..=2*size-1, offset by +1 into a
    // 0..=2*size array (max length 2*32+1 == 65).
    let mut above_row_buf = [0i32; 65];
    let above_row = &mut above_row_buf[..2 * size + 1];
    for i in 0..size {
        above_row[i + 1] = if have_above {
            let sx = (x + i).min(max_x);
            plane.get(sx, y - 1) as i32
        } else {
            base - 1
        };
    }
    for i in size..2 * size {
        above_row[i + 1] = if have_above && not_on_right && tx_size == TX_4X4 {
            let sx = (x + i).min(max_x);
            plane.get(sx, y - 1) as i32
        } else {
            above_row[size]
        };
    }
    above_row[0] = if have_above && have_left {
        let sx = x.saturating_sub(1).min(max_x);
        plane.get(sx, y - 1) as i32
    } else if have_above {
        base + 1
    } else {
        base - 1
    };
    // above_row[i+1] corresponds to the spec's aboveRow[i]; aboveRow[-1] is above_row[0].
    let above = |i: i32| -> i32 { above_row[(i + 1) as usize] };

    let mut left_col_buf = [0i32; 32];
    let left_col = &mut left_col_buf[..size];
    for (i, slot) in left_col.iter_mut().enumerate() {
        *slot = if have_left {
            let sy = (y + i).min(max_y);
            plane.get(x - 1, sy) as i32
        } else {
            base + 1
        };
    }

    let mut pred_buf = [0i32; 1024];
    let pred = &mut pred_buf[..size * size];
    let at = |i: usize, j: usize| i * size + j;

    match mode {
        V_PRED => {
            for i in 0..size {
                for j in 0..size {
                    pred[at(i, j)] = above(j as i32);
                }
            }
        }
        H_PRED => {
            for i in 0..size {
                for j in 0..size {
                    pred[at(i, j)] = left_col[i];
                }
            }
        }
        D207_PRED => {
            for j in 0..size {
                pred[at(size - 1, j)] = left_col[size - 1];
            }
            for i in 0..size.saturating_sub(1) {
                pred[at(i, 0)] = round2(left_col[i] + left_col[i + 1], 1);
            }
            for i in 0..size.saturating_sub(2) {
                pred[at(i, 1)] = round2(left_col[i] + 2 * left_col[i + 1] + left_col[i + 2], 2);
            }
            if size >= 2 {
                pred[at(size - 2, 1)] = round2(left_col[size - 2] + 3 * left_col[size - 1], 2);
            }
            if size >= 3 {
                for i in (0..=(size - 2)).rev() {
                    for j in 2..size {
                        pred[at(i, j)] = pred[at(i + 1, j - 2)];
                    }
                }
            }
        }
        D45_PRED => {
            for i in 0..size {
                for j in 0..size {
                    let s = (i + j) as i32;
                    pred[at(i, j)] = if s + 2 < (size as i32) * 2 {
                        round2(above(s) + above(s + 1) * 2 + above(s + 2), 2)
                    } else {
                        above(2 * size as i32 - 1)
                    };
                }
            }
        }
        D63_PRED => {
            for i in 0..size {
                for j in 0..size {
                    let half = (i / 2) as i32 + j as i32;
                    pred[at(i, j)] = if i & 1 == 1 {
                        round2(above(half) + above(half + 1) * 2 + above(half + 2), 2)
                    } else {
                        round2(above(half) + above(half + 1), 1)
                    };
                }
            }
        }
        D117_PRED => {
            for j in 0..size {
                pred[at(0, j)] = round2(above(j as i32 - 1) + above(j as i32), 1);
            }
            pred[at(1, 0)] = round2(left_col[0] + 2 * above(-1) + above(0), 2);
            for j in 1..size {
                pred[at(1, j)] = round2(
                    above(j as i32 - 2) + 2 * above(j as i32 - 1) + above(j as i32),
                    2,
                );
            }
            if size >= 3 {
                pred[at(2, 0)] = round2(above(-1) + 2 * left_col[0] + left_col[1], 2);
            }
            for i in 3..size {
                pred[at(i, 0)] = round2(left_col[i - 3] + 2 * left_col[i - 2] + left_col[i - 1], 2);
            }
            for i in 2..size {
                for j in 1..size {
                    pred[at(i, j)] = pred[at(i - 2, j - 1)];
                }
            }
        }
        D135_PRED => {
            pred[at(0, 0)] = round2(left_col[0] + 2 * above(-1) + above(0), 2);
            for j in 1..size {
                pred[at(0, j)] = round2(
                    above(j as i32 - 2) + 2 * above(j as i32 - 1) + above(j as i32),
                    2,
                );
            }
            if size >= 2 {
                pred[at(1, 0)] = round2(above(-1) + 2 * left_col[0] + left_col[1], 2);
            }
            for i in 2..size {
                pred[at(i, 0)] = round2(left_col[i - 2] + 2 * left_col[i - 1] + left_col[i], 2);
            }
            for i in 1..size {
                for j in 1..size {
                    pred[at(i, j)] = pred[at(i - 1, j - 1)];
                }
            }
        }
        D153_PRED => {
            pred[at(0, 0)] = round2(left_col[0] + above(-1), 1);
            for i in 1..size {
                pred[at(i, 0)] = round2(left_col[i - 1] + left_col[i], 1);
            }
            if size >= 2 {
                pred[at(0, 1)] = round2(left_col[0] + 2 * above(-1) + above(0), 2);
                pred[at(1, 1)] = round2(above(-1) + 2 * left_col[0] + left_col[1], 2);
            }
            for i in 2..size {
                pred[at(i, 1)] = round2(left_col[i - 2] + 2 * left_col[i - 1] + left_col[i], 2);
            }
            for j in 2..size {
                pred[at(0, j)] = round2(
                    above(j as i32 - 3) + 2 * above(j as i32 - 2) + above(j as i32 - 1),
                    2,
                );
            }
            for i in 1..size {
                for j in 2..size {
                    pred[at(i, j)] = pred[at(i - 1, j - 2)];
                }
            }
        }
        TM_PRED => {
            let top_left = above(-1);
            for i in 0..size {
                for j in 0..size {
                    pred[at(i, j)] = clip3(
                        0,
                        (1 << bit_depth) - 1,
                        above(j as i32) + left_col[i] - top_left,
                    );
                }
            }
        }
        DC_PRED => {
            let value = if have_left && have_above {
                let mut sum = 0i32;
                for (k, &l) in left_col.iter().enumerate() {
                    sum += l;
                    sum += above(k as i32);
                }
                (sum + size as i32) >> (log2_size + 1)
            } else if have_left {
                let sum: i32 = left_col.iter().sum();
                (sum + (1 << (log2_size - 1))) >> log2_size
            } else if have_above {
                let sum: i32 = (0..size).map(|k| above(k as i32)).sum();
                (sum + (1 << (log2_size - 1))) >> log2_size
            } else {
                base
            };
            pred.fill(value);
        }
        _ => unreachable!("predict_intra: unknown intra prediction mode {mode}"),
    }

    for i in 0..size {
        for j in 0..size {
            plane.set(x + j, y + i, pred[at(i, j)] as u16);
        }
    }
}

/// A view over one plane of a reference frame (equivalent to the spec's
/// `FrameStore[ refIdx ][ plane ]` + `RefFrameWidth`/`RefFrameHeight`).
/// `width`/`height` are the overall frame size based on luma (plane 0) — this
/// is exactly the `RefFrameWidth[refIdx]` used in the scaling computation of
/// spec §8.5.2.3 (pass the luma size even for chroma planes). `plane` is the
/// reference pixel data already cropped to the relevant plane (`Plane::width`/
/// `height` are directly the spec's `lastX+1`/`lastY+1`).
pub struct RefPlaneView<'a> {
    pub plane: &'a Plane,
    pub width: u32,
    pub height: u32,
}

#[inline]
fn round_mv_comp_q2(value: i32) -> i32 {
    if value < 0 {
        (value - 1) / 2
    } else {
        (value + 1) / 2
    }
}

#[inline]
fn round_mv_comp_q4(value: i32) -> i32 {
    if value < 0 {
        (value - 2) / 4
    } else {
        (value + 2) / 4
    }
}

/// `clip1(x)` = `Clip3( 0, (1<<BitDepth)-1, x )` (spec §4).
#[inline]
fn clip1(x: i32, bit_depth: u8) -> i32 {
    clip3(0, (1i32 << bit_depth) - 1, x)
}

/// Per-block-invariant parameters for `predict_inter` (spec §8.5.2) and its helpers
/// (`select_mv`/`clamp_mv_for_plane`/`scale_mv_for_plane`).
///
/// A single coding block's `residual()` (`src/tile/residual.rs`) calls `predict_inter` once
/// per plane (further split into 4x4 sub-blocks when `MiSize < BLOCK_8X8`); every field here
/// is identical across all of those calls -- only `dst`/`plane`/`x`/`y`/`w`/`h`/`block_idx`
/// (`predict_inter`'s remaining positional parameters) vary call to call.
pub struct InterPredictParams<'a> {
    pub ref_frame: [u8; 2],
    pub block_mvs: &'a [[Mv; 4]; 2],
    pub interp_filter: u8,
    pub mi_row: u32,
    pub mi_col: u32,
    pub mi_size: u8,
    pub mi_rows: u32,
    pub mi_cols: u32,
    pub subsampling_x: u32,
    pub subsampling_y: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub bit_depth: u8,
    pub refs: [Option<&'a RefPlaneView<'a>>; 2],
}

/// Motion vector selection process (spec §8.5.2.1 "Motion vector selection process").
fn select_mv(plane: usize, ref_list: usize, block_idx: usize, p: &InterPredictParams) -> Mv {
    let bm = &p.block_mvs[ref_list];
    if plane == 0 || p.mi_size >= BLOCK_8X8 {
        return bm[block_idx];
    }
    match (p.subsampling_x, p.subsampling_y) {
        (0, 0) => bm[block_idx],
        (0, 1) => [
            round_mv_comp_q2(bm[block_idx][0] + bm[block_idx + 2][0]),
            round_mv_comp_q2(bm[block_idx][1] + bm[block_idx + 2][1]),
        ],
        (1, 0) => [
            round_mv_comp_q2(bm[block_idx][0] + bm[block_idx + 1][0]),
            round_mv_comp_q2(bm[block_idx][1] + bm[block_idx + 1][1]),
        ],
        _ => [
            round_mv_comp_q4(bm[0][0] + bm[1][0] + bm[2][0] + bm[3][0]),
            round_mv_comp_q4(bm[0][1] + bm[1][1] + bm[2][1] + bm[3][1]),
        ],
    }
}

/// Motion vector clamping process (spec §8.5.2.2 "Motion vector clamping process").
fn clamp_mv_for_plane(plane: usize, mv: Mv, p: &InterPredictParams) -> Mv {
    let (sx, sy) = if plane == 0 {
        (0i32, 0i32)
    } else {
        (p.subsampling_x as i32, p.subsampling_y as i32)
    };
    let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[p.mi_size as usize] as i32;
    let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[p.mi_size as usize] as i32;
    let mi_row = p.mi_row as i32;
    let mi_col = p.mi_col as i32;
    let mi_rows = p.mi_rows as i32;
    let mi_cols = p.mi_cols as i32;

    let mb_to_top_edge = -((mi_row * 8) * 16) >> sy;
    let mb_to_bottom_edge = ((mi_rows - bh - mi_row) * 8 * 16) >> sy;
    let mb_to_left_edge = -((mi_col * 8) * 16) >> sx;
    let mb_to_right_edge = ((mi_cols - bw - mi_col) * 8 * 16) >> sx;

    let spel_left = (INTERP_EXTEND + ((bw * 8) >> sx)) << SUBPEL_BITS;
    let spel_right = spel_left - SUBPEL_SHIFTS;
    let spel_top = (INTERP_EXTEND + ((bh * 8) >> sy)) << SUBPEL_BITS;
    let spel_bottom = spel_top - SUBPEL_SHIFTS;

    [
        clip3(
            mb_to_top_edge - spel_top,
            mb_to_bottom_edge + spel_bottom,
            (2 * mv[0]) >> sy,
        ),
        clip3(
            mb_to_left_edge - spel_left,
            mb_to_right_edge + spel_right,
            (2 * mv[1]) >> sx,
        ),
    ]
}

/// Motion vector scaling process (spec §8.5.2.3 "Motion vector scaling process").
/// Returns `(startX, startY, stepX, stepY)` (in 1/16 pel units). `ref_width`/`ref_height` are
/// per-call (the specific reference's own size, `view.width`/`view.height` at the call site),
/// unlike the rest of `p`.
fn scale_mv_for_plane(
    plane: usize,
    x: usize,
    y: usize,
    clamped_mv: Mv,
    ref_width: u32,
    ref_height: u32,
    p: &InterPredictParams,
) -> (i64, i64, i64, i64) {
    // Fast path for the common unscaled case (same-size reference): x_scale == y_scale == 16384
    // (1 << REF_SCALE_SHIFT), so base = (x, y), frac = (0, 0), d = clamped_mv, step = (16, 16).
    if ref_width == p.frame_width && ref_height == p.frame_height {
        return (
            ((x as i64) << SUBPEL_BITS) + clamped_mv[1] as i64,
            ((y as i64) << SUBPEL_BITS) + clamped_mv[0] as i64,
            16,
            16,
        );
    }
    let x_scale = ((ref_width as i64) << REF_SCALE_SHIFT) / p.frame_width as i64;
    let y_scale = ((ref_height as i64) << REF_SCALE_SHIFT) / p.frame_height as i64;
    let base_x = ((x as i64) * x_scale) >> REF_SCALE_SHIFT;
    let base_y = ((y as i64) * y_scale) >> REF_SCALE_SHIFT;
    let luma_x = if plane > 0 {
        (x as u32) << p.subsampling_x
    } else {
        x as u32
    } as i64;
    let luma_y = if plane > 0 {
        (y as u32) << p.subsampling_y
    } else {
        y as u32
    } as i64;
    let frac_x = ((16 * luma_x * x_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK as i64;
    let frac_y = ((16 * luma_y * y_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK as i64;
    let d_x = ((clamped_mv[1] as i64 * x_scale) >> REF_SCALE_SHIFT) + frac_x;
    let d_y = ((clamped_mv[0] as i64 * y_scale) >> REF_SCALE_SHIFT) + frac_y;
    let step_x = (16 * x_scale) >> REF_SCALE_SHIFT;
    let step_y = (16 * y_scale) >> REF_SCALE_SHIFT;
    let start_x = (base_x << SUBPEL_BITS) + d_x;
    let start_y = (base_y << SUBPEL_BITS) + d_y;
    (start_x, start_y, step_x, step_y)
}

/// Max output block dimension for a single `predict_inter` call (spec: 64x64 is the
/// largest coding block, and chroma calls are always <= that due to subsampling).
/// `pub(crate)`: shared with `src/simd/inter.rs`'s AVX2 mirror of [`block_inter_predict`].
pub(crate) const MAX_BLOCK_DIM: usize = 64;

/// Max rows needed in [`block_inter_predict`]'s intermediate (horizontal-filter) buffer.
/// `h <= MAX_BLOCK_DIM`, and the vertical step `y_step` is bounded by the spec's reference-
/// scaling conformance requirement (§8.5.2.3: `RefFrameHeight <= 2 * FrameHeight`), which
/// caps `y_step <= 32` (1/16-pel units; see [`scale_mv_for_plane`]); the 8-tap subpel filter
/// needs 8 extra rows of context. `(((MAX_BLOCK_DIM - 1) * 32 + 15) >> 4) + 8 == 134`.
/// The 2x bound is *enforced*, not assumed: `decode_block` (`src/tile.rs`) rejects any block
/// referencing a frame beyond it (`TileError::RefFrameSizeOutOfRange`), so decode-path
/// callers never exceed this scratch size on any input, conformant or not.
/// `pub(crate)`: shared with `src/simd/inter.rs`'s AVX2 mirror of [`block_inter_predict`].
pub(crate) const MAX_INTERMEDIATE_HEIGHT: usize = 134;

/// Scalar body of [`block_inter_predict_into_plane`] (spec §8.5.2.4 two-pass loops): the
/// always-kept fallback for every case the AVX2 kernels don't take (`VP9DEC_NO_SIMD=1`,
/// non-x86_64, or an unsupported dispatch shape), and the bit-exactness oracle
/// `tests/unit/simd.rs`'s unit tests pin the kernels against (hence `pub(crate)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_inter_predict_scalar(
    ref_plane: &Plane,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    interp_filter: u8,
    bit_depth: u8,
    pred: &mut [i32],
) {
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert_eq!(pred.len(), w * h);
    let last_x = ref_plane.width as i64 - 1;
    let last_y = ref_plane.height as i64 - 1;
    let intermediate_height = (((h as i64 - 1) * y_step + 15) >> 4) + 8;
    debug_assert!(intermediate_height as usize <= MAX_INTERMEDIATE_HEIGHT);

    let filters = &SUBPEL_FILTERS[interp_filter as usize];
    let ref_data = ref_plane.as_slice();
    let ref_w = ref_plane.width;
    let ref_stride = ref_plane.stride();

    // Fast unscaled scalar paths: avoid running zero taps when fx == 0 or fy == 0, and avoid
    // 34 KB stack zeroing on every small block.
    if x_step == 16 && y_step == 16 {
        let fx = (x & 15) as usize;
        let fy = (y & 15) as usize;
        let x0 = x >> 4;
        let y0 = y >> 4;
        let max_val = (1i32 << bit_depth) - 1;

        if fx == 0 && fy == 0 {
            for r in 0..h {
                let ref_y = (y0 + r as i64).clamp(0, last_y) as usize;
                let row = &ref_data[ref_y * ref_stride..ref_y * ref_stride + ref_w];
                if x0 >= 0 && x0 + w as i64 - 1 <= last_x {
                    let sx = x0 as usize;
                    for c in 0..w {
                        pred[r * w + c] = row[sx + c] as i32;
                    }
                } else {
                    for c in 0..w {
                        let sx = (x0 + c as i64).clamp(0, last_x) as usize;
                        pred[r * w + c] = row[sx] as i32;
                    }
                }
            }
            return;
        }
        if fy == 0 {
            let hcoeffs = &filters[fx];
            let in_x = x0 - 3 >= 0 && x0 + w as i64 + 3 <= last_x;
            for r in 0..h {
                let ref_y = (y0 + r as i64).clamp(0, last_y) as usize;
                let row = &ref_data[ref_y * ref_stride..ref_y * ref_stride + ref_w];
                if in_x {
                    let sx0 = (x0 - 3) as usize;
                    for c in 0..w {
                        let base = &row[sx0 + c..sx0 + c + 8];
                        let s = hcoeffs[0] * base[0] as i32
                            + hcoeffs[1] * base[1] as i32
                            + hcoeffs[2] * base[2] as i32
                            + hcoeffs[3] * base[3] as i32
                            + hcoeffs[4] * base[4] as i32
                            + hcoeffs[5] * base[5] as i32
                            + hcoeffs[6] * base[6] as i32
                            + hcoeffs[7] * base[7] as i32;
                        pred[r * w + c] = clip3(0, max_val, round2(s, 7));
                    }
                } else {
                    for c in 0..w {
                        let mut s = 0i32;
                        for (t, &coeff) in hcoeffs.iter().enumerate() {
                            let sx = (x0 + c as i64 + t as i64 - 3).clamp(0, last_x) as usize;
                            s += coeff * row[sx] as i32;
                        }
                        pred[r * w + c] = clip3(0, max_val, round2(s, 7));
                    }
                }
            }
            return;
        }
        if fx == 0 {
            let vcoeffs = &filters[fy];
            let in_x = x0 >= 0 && x0 + w as i64 - 1 <= last_x;
            let in_y = y0 - 3 >= 0 && y0 + h as i64 + 3 <= last_y;
            if in_x && in_y {
                let sx = x0 as usize;
                for r in 0..h {
                    let ybase = (y0 - 3) as usize + r;
                    let r0 = &ref_data[ybase * ref_stride + sx..ybase * ref_stride + sx + w];
                    let r1 =
                        &ref_data[(ybase + 1) * ref_stride + sx..(ybase + 1) * ref_stride + sx + w];
                    let r2 =
                        &ref_data[(ybase + 2) * ref_stride + sx..(ybase + 2) * ref_stride + sx + w];
                    let r3 =
                        &ref_data[(ybase + 3) * ref_stride + sx..(ybase + 3) * ref_stride + sx + w];
                    let r4 =
                        &ref_data[(ybase + 4) * ref_stride + sx..(ybase + 4) * ref_stride + sx + w];
                    let r5 =
                        &ref_data[(ybase + 5) * ref_stride + sx..(ybase + 5) * ref_stride + sx + w];
                    let r6 =
                        &ref_data[(ybase + 6) * ref_stride + sx..(ybase + 6) * ref_stride + sx + w];
                    let r7 =
                        &ref_data[(ybase + 7) * ref_stride + sx..(ybase + 7) * ref_stride + sx + w];
                    for c in 0..w {
                        let s = vcoeffs[0] * r0[c] as i32
                            + vcoeffs[1] * r1[c] as i32
                            + vcoeffs[2] * r2[c] as i32
                            + vcoeffs[3] * r3[c] as i32
                            + vcoeffs[4] * r4[c] as i32
                            + vcoeffs[5] * r5[c] as i32
                            + vcoeffs[6] * r6[c] as i32
                            + vcoeffs[7] * r7[c] as i32;
                        pred[r * w + c] = clip3(0, max_val, round2(s, 7));
                    }
                }
                return;
            }
        }
    }

    // General scalar path (scaled or 2D unscaled).
    let mut intermediate = [0i32; MAX_INTERMEDIATE_HEIGHT * MAX_BLOCK_DIM];
    for r in 0..intermediate_height {
        let ref_y = ((y >> 4) + r - 3).clamp(0, last_y) as usize;
        let row = &ref_data[ref_y * ref_stride..ref_y * ref_stride + ref_w];
        for c in 0..w {
            let p = x + x_step * (c as i64);
            let coeffs = &filters[(p & 15) as usize];
            let mut s = 0i32;
            for (t, &coeff) in coeffs.iter().enumerate() {
                let ref_x = ((p >> 4) + t as i64 - 3).clamp(0, last_x) as usize;
                s += coeff * row[ref_x] as i32;
            }
            intermediate[(r as usize) * w + c] = clip1(round2(s, 7), bit_depth);
        }
    }

    for r in 0..h {
        let p = (y & 15) + y_step * (r as i64);
        let coeffs = &filters[(p & 15) as usize];
        let base_row = (p >> 4) as usize;
        for c in 0..w {
            let mut s = 0i32;
            for (t, &coeff) in coeffs.iter().enumerate() {
                s += coeff * intermediate[(base_row + t) * w + c];
            }
            pred[r * w + c] = clip1(round2(s, 7), bit_depth);
        }
    }
}

/// Writes one reference list's inter prediction directly into `dst` (or averages it with `dst`
/// when `compound` is true). Dispatches to the direct-to-u16 AVX2 kernels on `x86_64`, avoiding
/// both the 32 KB `preds` stack zeroing and the scalar `i32 -> u16` copy-out loop.
#[allow(clippy::too_many_arguments)]
fn block_inter_predict_into_plane(
    ref_plane: &Plane,
    dst: &mut Plane,
    dst_x: usize,
    dst_y: usize,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    interp_filter: u8,
    bit_depth: u8,
    compound: bool,
) {
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert!(dst_x >= dst.x0 && dst_x - dst.x0 + w <= dst.width);
    debug_assert!(dst_y + h <= dst.height);

    #[cfg(target_arch = "x86_64")]
    if (w == 4 || w.is_multiple_of(8)) && crate::simd::avx2_enabled() {
        let max_val = (1i32 << bit_depth) - 1;
        let local_x = dst_x - dst.x0;
        let dst_stride = dst.stride();
        let ref_stride = ref_plane.stride();
        let last_x = ref_plane.width as i64 - 1;
        let last_y = ref_plane.height as i64 - 1;

        if x_step == 16 && y_step == 16 {
            let x0 = x >> 4;
            let y0 = y >> 4;
            let fx = (x & 15) as usize;
            let fy = (y & 15) as usize;
            let x_ok = if fx == 0 {
                x0 >= 0 && x0 + (w as i64 - 1) <= last_x
            } else {
                x0 - 3 >= 0 && x0 + (w as i64 - 1) + 4 <= last_x
            };
            let y_ok = if fy == 0 {
                y0 >= 0 && y0 + (h as i64 - 1) <= last_y
            } else {
                y0 - 3 >= 0 && y0 + (h as i64 - 1) + 4 <= last_y
            };
            if x_ok && y_ok {
                // SAFETY: `avx2_enabled()` confirmed; `x_ok && y_ok` proves every source pixel
                // read by the active `(fx, fy)` branch lies in `ref_plane`, and `dst` bounds hold.
                unsafe {
                    if compound {
                        crate::simd::block_inter_predict_u16_avx2::<true>(
                            ref_plane.as_slice(),
                            ref_stride,
                            dst.as_mut_slice(),
                            dst_stride,
                            local_x,
                            dst_y,
                            x0,
                            y0,
                            fx,
                            fy,
                            w,
                            h,
                            interp_filter,
                            max_val,
                        );
                    } else {
                        crate::simd::block_inter_predict_u16_avx2::<false>(
                            ref_plane.as_slice(),
                            ref_stride,
                            dst.as_mut_slice(),
                            dst_stride,
                            local_x,
                            dst_y,
                            x0,
                            y0,
                            fx,
                            fy,
                            w,
                            h,
                            interp_filter,
                            max_val,
                        );
                    }
                }
                return;
            }
        }

        let intermediate_height = (((h as i64 - 1) * y_step + 15) >> 4) + 8;
        let span = (((x & 15) + x_step * (w as i64 - 1)) >> 4) + 8;
        if intermediate_height as usize <= MAX_INTERMEDIATE_HEIGHT
            && span as usize <= MAX_INTERMEDIATE_HEIGHT
        {
            // SAFETY: `avx2_enabled()` confirmed; `ref_plane` has `stride * height` samples,
            // every source read is clamped inside `width x height`, scratch bounds were checked,
            // and `dst` bounds hold.
            unsafe {
                if compound {
                    crate::simd::block_inter_predict_scaled_u16_avx2::<true>(
                        ref_plane.as_slice(),
                        ref_plane.width,
                        ref_stride,
                        ref_plane.height,
                        dst.as_mut_slice(),
                        dst_stride,
                        local_x,
                        dst_y,
                        x,
                        y,
                        x_step,
                        y_step,
                        w,
                        h,
                        intermediate_height as usize,
                        interp_filter,
                        max_val,
                    );
                } else {
                    crate::simd::block_inter_predict_scaled_u16_avx2::<false>(
                        ref_plane.as_slice(),
                        ref_plane.width,
                        ref_stride,
                        ref_plane.height,
                        dst.as_mut_slice(),
                        dst_stride,
                        local_x,
                        dst_y,
                        x,
                        y,
                        x_step,
                        y_step,
                        w,
                        h,
                        intermediate_height as usize,
                        interp_filter,
                        max_val,
                    );
                }
            }
            return;
        }
    }

    // Scalar fallback (non-x86_64 or `VP9DEC_NO_SIMD=1`).
    let mut pred = [0i32; MAX_BLOCK_DIM * MAX_BLOCK_DIM];
    block_inter_predict_scalar(
        ref_plane,
        x,
        y,
        x_step,
        y_step,
        w,
        h,
        interp_filter,
        bit_depth,
        &mut pred[..w * h],
    );
    let local_x = dst_x - dst.x0;
    let dst_stride = dst.stride();
    let dst_slice = dst.as_mut_slice();
    if compound {
        for i in 0..h {
            let dst_row = &mut dst_slice[(dst_y + i) * dst_stride + local_x..][..w];
            let src_row = &pred[i * w..(i + 1) * w];
            for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                *d = round2(*d as i32 + s, 1) as u16;
            }
        }
    } else {
        for i in 0..h {
            let dst_row = &mut dst_slice[(dst_y + i) * dst_stride + local_x..][..w];
            let src_row = &pred[i * w..(i + 1) * w];
            for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                *d = s as u16;
            }
        }
    }
}

/// `predict_inter()` (spec §8.5.2 "Inter prediction process").
///
/// Writes motion-compensated, sub-pixel-interpolated inter prediction
/// samples into the `w x h` region of `dst` with top-left corner `(x, y)`.
/// `block_idx` is the spec's `blockIdx` (in 4x4 units, representing how much
/// of this block has been predicted so far; only meaningful when
/// `MiSize < BLOCK_8X8`). `p` groups the per-block-invariant parameters (see
/// [`InterPredictParams`]); still `#[allow(too_many_arguments)]` at 8 (down from the
/// pre-W5 20).
#[allow(clippy::too_many_arguments)]
pub fn predict_inter(
    dst: &mut Plane,
    plane: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    block_idx: usize,
    p: &InterPredictParams,
) {
    use crate::prob_tables::INTRA_FRAME;
    let is_compound = p.ref_frame[1] > INTRA_FRAME;
    let n_refs = 1 + is_compound as usize;

    for ref_list in 0..n_refs {
        let mv = select_mv(plane, ref_list, block_idx, p);
        let clamped = clamp_mv_for_plane(plane, mv, p);
        let view = p.refs[ref_list].expect("reference frame slot is missing (DPB not initialized)");
        let (start_x, start_y, step_x, step_y) =
            scale_mv_for_plane(plane, x, y, clamped, view.width, view.height, p);
        block_inter_predict_into_plane(
            view.plane,
            dst,
            x,
            y,
            start_x,
            start_y,
            step_x,
            step_y,
            w,
            h,
            p.interp_filter,
            p.bit_depth,
            ref_list == 1,
        );
    }
}

#[cfg(test)]
#[path = "../tests/unit/predict.rs"]
mod tests;
