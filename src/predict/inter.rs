//! Inter prediction / motion compensation process (spec §8.5.2): MV selection, edge clamping,
//! reference-frame scaling (§8.5.2.1-§8.5.2.3), two-pass 8-tap subpel interpolation (§8.5.2.4,
//! with direct-to-u16 AVX2 fast paths), and compound averaging.

use crate::common::{clip3, round2};
use crate::framebuffer::Plane;
use crate::prob_tables::{
    BLOCK_8X8, INTRA_FRAME, NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP,
};
use crate::subpel::{
    INTERP_EXTEND, REF_SCALE_SHIFT, SUBPEL_BITS, SUBPEL_FILTERS, SUBPEL_MASK, SUBPEL_SHIFTS,
};
use crate::tile::Mv;

/// A view over one plane of a reference frame (equivalent to the spec's
/// `FrameStore[ refIdx ][ plane ]` + `RefFrameWidth`/`RefFrameHeight`).
/// `width`/`height` are the overall frame size based on luma (plane 0) -- the
/// `RefFrameWidth[refIdx]` used in spec §8.5.2.3's scaling computation. `plane` is the
/// reference pixel data cropped to the relevant plane (`Plane::width`/`height` are `lastX+1`/`lastY+1`).
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

/// Per-block-invariant parameters for [`predict_inter`] (spec §8.5.2) and its helpers.
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

/// Motion vector selection process (spec §8.5.2.1).
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

/// Motion vector clamping process (spec §8.5.2.2).
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

/// Motion vector scaling process (spec §8.5.2.3). Returns `(startX, startY, stepX, stepY)`
/// in 1/16-pel units.
fn scale_mv_for_plane(
    plane: usize,
    x: usize,
    y: usize,
    clamped_mv: Mv,
    ref_width: u32,
    ref_height: u32,
    p: &InterPredictParams,
) -> (i64, i64, i64, i64) {
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

/// Max output block dimension for a single `predict_inter` call (64x64).
pub(crate) const MAX_BLOCK_DIM: usize = 64;

/// Max rows needed in the intermediate (horizontal-filter) buffer under spec §8.5.2.3's 2x
/// reference-scaling bound: `(((MAX_BLOCK_DIM - 1) * 32 + 15) >> 4) + 8 == 134`.
pub(crate) const MAX_INTERMEDIATE_HEIGHT: usize = 134;

/// Scalar body of [`block_inter_predict_into_plane`] (spec §8.5.2.4 two-pass loops): the
/// always-kept fallback and bit-exactness oracle `tests/unit/simd.rs` pins AVX2 against.
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
/// when `compound` is true). Dispatches to direct-to-u16 AVX2 kernels on `x86_64`.
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
    if (w == 4 || w.is_multiple_of(8))
        && crate::simd::avx2_enabled()
        && try_inter_predict_avx2(
            ref_plane,
            dst,
            dst_x,
            dst_y,
            x,
            y,
            x_step,
            y_step,
            w,
            h,
            interp_filter,
            bit_depth,
            compound,
        )
    {
        return;
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

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn try_inter_predict_avx2(
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
) -> bool {
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
            // SAFETY: `avx2_enabled()` confirmed by caller; `x_ok && y_ok` proves every source
            // pixel read lies in `ref_plane`, and `dst` bounds hold.
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
            return true;
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
        return true;
    }
    false
}

/// `predict_inter()` (spec §8.5.2 "Inter prediction process").
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
