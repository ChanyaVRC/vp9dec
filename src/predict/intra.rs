//! Intra prediction process (spec §8.5.1): the 10 VP9 intra modes (`DC_PRED` through `TM_PRED`),
//! with runtime-detected AVX2 dispatch for the high-frequency modes (`V_PRED`, `H_PRED`,
//! `DC_PRED`, `TM_PRED`) across all transform sizes (4/8/16/32) and bit depths (8/10/12).

use crate::common::{clip3, round2};
use crate::framebuffer::Plane;
use crate::prob_tables::{
    D117_PRED, D135_PRED, D153_PRED, D207_PRED, D45_PRED, D63_PRED, DC_PRED, H_PRED, TM_PRED,
    TX_4X4, V_PRED,
};

#[cfg(target_arch = "x86_64")]
#[inline]
fn load_clamped_above(plane: &Plane, x: usize, y: usize, max_x: usize, dst: &mut [u16]) {
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = plane.get((x + i).min(max_x), y - 1);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn load_clamped_left(plane: &Plane, x: usize, y: usize, max_y: usize, dst: &mut [u16]) {
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = plane.get(x - 1, (y + i).min(max_y));
    }
}

/// Spec §8.5.1's `predict_intra` process.
///
/// `x`/`y` are absolute pixel coordinates within `plane`; `max_x`/`max_y` are
/// the spec's `maxX`/`maxY` (= the maximum coordinates that may be
/// referenced within the plane).
///
/// Dispatches `V_PRED`, `H_PRED`, `DC_PRED`, and `TM_PRED` to AVX2 when available; falls back
/// to `predict_intra_scalar` for the six directional modes, `VP9DEC_NO_SIMD=1`, and
/// non-x86_64 targets.
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
        predict_intra_avx2(
            plane, x, y, have_left, have_above, tx_size, mode, max_x, max_y, bit_depth,
        );
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

/// Runs `f(dst_rows, dst_y, above)` borrowing the above row directly via `split_at_mut` when
/// `have_above && x + size - 1 <= max_x`, or via a stack-clamped buffer otherwise.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn with_above_row<F: FnOnce(&mut [u16], usize, Option<&[u16]>)>(
    plane: &mut Plane,
    x: usize,
    y: usize,
    size: usize,
    max_x: usize,
    have_above: bool,
    f: F,
) {
    let pw = plane.stride();
    let local_x = x - plane.x0;
    if have_above && x + size - 1 <= max_x {
        let (above_rows, dst_rows) = plane.as_mut_slice().split_at_mut(y * pw);
        let above_start = (y - 1) * pw + local_x;
        f(
            dst_rows,
            0,
            Some(&above_rows[above_start..above_start + size]),
        );
    } else if have_above {
        let mut above_buf = [0u16; 32];
        load_clamped_above(plane, x, y, max_x, &mut above_buf[..size]);
        f(plane.as_mut_slice(), y, Some(&above_buf[..size]));
    } else {
        f(plane.as_mut_slice(), y, None);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn predict_intra_avx2(
    plane: &mut Plane,
    x: usize,
    y: usize,
    have_left: bool,
    have_above: bool,
    tx_size: u8,
    mode: u8,
    max_x: usize,
    max_y: usize,
    bit_depth: u8,
) {
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
                // SAFETY: avx2_enabled() confirmed by caller; block bounds checked above.
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
            } else {
                with_above_row(plane, x, y, size, max_x, true, |dst, dy, above| {
                    // SAFETY: avx2_enabled() confirmed; `above` has len `size` and bounds hold.
                    unsafe {
                        crate::simd::intra_pred_v_avx2(dst, pw, local_x, dy, above.unwrap(), size);
                    }
                });
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
                load_clamped_left(plane, x, y, max_y, &mut left_buf[..size]);
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
            let left_opt = have_left.then(|| {
                load_clamped_left(plane, x, y, max_y, &mut left_buf[..size]);
                &left_buf[..size]
            });
            with_above_row(
                plane,
                x,
                y,
                size,
                max_x,
                have_above,
                |dst, dy, above_opt| {
                    // SAFETY: avx2_enabled() confirmed; neighbor slices and block bounds hold.
                    unsafe {
                        crate::simd::intra_pred_dc_avx2(
                            dst, pw, local_x, dy, above_opt, left_opt, log2_size, bit_depth,
                        );
                    }
                },
            );
        }
        TM_PRED => {
            let top_left = if have_above && have_left {
                plane.get(x.saturating_sub(1).min(max_x), y - 1)
            } else if have_above {
                base + 1
            } else {
                base - 1
            };
            let mut left_buf = [base + 1; 32];
            if have_left {
                load_clamped_left(plane, x, y, max_y, &mut left_buf[..size]);
            }
            let fallback_above = [base - 1; 32];
            with_above_row(
                plane,
                x,
                y,
                size,
                max_x,
                have_above,
                |dst, dy, above_opt| {
                    let above = above_opt.unwrap_or(&fallback_above[..size]);
                    // SAFETY: avx2_enabled() confirmed; neighbor slices and block bounds hold.
                    unsafe {
                        crate::simd::intra_pred_tm_avx2(
                            dst,
                            pw,
                            local_x,
                            dy,
                            above,
                            &left_buf[..size],
                            top_left,
                            size,
                            bit_depth,
                        );
                    }
                },
            );
        }
        _ => unreachable!(),
    }
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

    // above_row holds spec indices -1..=2*size-1 offset by +1 (max length 2*32 + 1 == 65).
    let mut above_row_buf = [0i32; 65];
    let above_row = &mut above_row_buf[..2 * size + 1];
    for i in 0..size {
        above_row[i + 1] = if have_above {
            plane.get((x + i).min(max_x), y - 1) as i32
        } else {
            base - 1
        };
    }
    for i in size..2 * size {
        above_row[i + 1] = if have_above && not_on_right && tx_size == TX_4X4 {
            plane.get((x + i).min(max_x), y - 1) as i32
        } else {
            above_row[size]
        };
    }
    above_row[0] = if have_above && have_left {
        plane.get(x.saturating_sub(1).min(max_x), y - 1) as i32
    } else if have_above {
        base + 1
    } else {
        base - 1
    };
    let above = |i: i32| -> i32 { above_row[(i + 1) as usize] };

    let mut left_col_buf = [0i32; 32];
    let left_col = &mut left_col_buf[..size];
    for (i, slot) in left_col.iter_mut().enumerate() {
        *slot = if have_left {
            plane.get(x - 1, (y + i).min(max_y)) as i32
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
