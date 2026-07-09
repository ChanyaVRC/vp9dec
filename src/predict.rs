//! イントラ予測処理（仕様 8.5.1 節 "Intra prediction process"）。
//!
//! VP9 のイントラ予測は 10 モード（`DC_PRED`〜`TM_PRED`）を持つ。VP8 以前や AV1 と異なり
//! smooth 系フィルタは存在しないため、ここではその 10 モードのみを実装する。
//!
//! 呼び出し側（[`crate::tile::TileDecoder`]）は変換ブロック単位で以下を渡す:
//! - `plane` の該当領域を含む [`crate::framebuffer::Plane`]（予測結果はここに書き込む）
//! - 予測対象ブロックの左上座標 `(x, y)`
//! - `have_left`/`have_above`/`not_on_right`（フレーム端・ブロック端の可用性）
//! - `tx_size`（`TX_4X4`=0 〜 `TX_32X32`=3）
//! - イントラ予測モード（`DC_PRED`〜`TM_PRED`）
//! - クリップ境界 `max_x`/`max_y`（= 仕様の `maxX`/`maxY`、プレーンごとに異なる）

use crate::framebuffer::Plane;
use crate::prob_tables::{
    D117_PRED, D135_PRED, D153_PRED, D207_PRED, D45_PRED, D63_PRED, DC_PRED, H_PRED, TM_PRED,
    TX_4X4, V_PRED,
};

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

/// 仕様 8.5.1 節の `predict_intra` プロセス。
///
/// `x`/`y` は `plane` 内の絶対ピクセル座標、`max_x`/`max_y` は仕様の `maxX`/`maxY`
/// （= プレーン内で参照してよい最大座標）。
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
    let log2_size = (tx_size as u32) + 2;
    let size = 1usize << log2_size;
    let base = 1i32 << (bit_depth - 1);

    // aboveRow は仕様の添字 -1..=2*size-1 を、+1 オフセットして 0..=2*size の配列で持つ。
    let mut above_row = vec![0i32; 2 * size + 1];
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
    // above_row[i+1] は仕様の aboveRow[i] に対応する。aboveRow[-1] は above_row[0]。
    let above = |i: i32| -> i32 { above_row[(i + 1) as usize] };

    let mut left_col = vec![0i32; size];
    for (i, slot) in left_col.iter_mut().enumerate() {
        *slot = if have_left {
            let sy = (y + i).min(max_y);
            plane.get(x - 1, sy) as i32
        } else {
            base + 1
        };
    }

    let mut pred = vec![0i32; size * size];
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
        _ => unreachable!("predict_intra: 未知のイントラ予測モード {mode}"),
    }

    for i in 0..size {
        for j in 0..size {
            plane.set(x + j, y + i, pred[at(i, j)] as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prob_tables::TX_4X4 as TX4;

    fn make_plane(width: usize, height: usize, fill: u8) -> Plane {
        let mut p = Plane::new(width, height);
        for y in 0..height {
            for x in 0..width {
                p.set(x, y, fill);
            }
        }
        p
    }

    #[test]
    fn dc_pred_with_no_neighbors_uses_bit_depth_midpoint() {
        let mut plane = Plane::new(8, 8);
        predict_intra(&mut plane, 0, 0, false, false, false, TX4, DC_PRED, 7, 7, 8);
        assert_eq!(plane.get(0, 0), 128);
        assert_eq!(plane.get(3, 3), 128);
    }

    #[test]
    fn v_pred_copies_above_row() {
        let mut plane = make_plane(8, 8, 0);
        for x in 0..4 {
            plane.set(x, 3, 50 + x as u8);
        }
        predict_intra(&mut plane, 0, 4, false, true, false, TX4, V_PRED, 7, 7, 8);
        for x in 0..4 {
            assert_eq!(plane.get(x, 4), 50 + x as u8);
            assert_eq!(plane.get(x, 5), 50 + x as u8);
        }
    }

    #[test]
    fn h_pred_copies_left_col() {
        let mut plane = make_plane(8, 8, 0);
        for y in 0..4 {
            plane.set(3, y, 60 + y as u8);
        }
        predict_intra(&mut plane, 4, 0, true, false, false, TX4, H_PRED, 7, 7, 8);
        for y in 0..4 {
            assert_eq!(plane.get(4, y), 60 + y as u8);
            assert_eq!(plane.get(5, y), 60 + y as u8);
        }
    }

    #[test]
    fn tm_pred_clips_to_valid_range() {
        let mut plane = make_plane(8, 8, 0);
        for x in 0..4 {
            plane.set(4 + x, 3, 255);
        }
        for y in 0..4 {
            plane.set(3, 4 + y, 255);
        }
        plane.set(3, 3, 0);
        predict_intra(&mut plane, 4, 4, true, true, false, TX4, TM_PRED, 7, 7, 8);
        // 255 + 255 - 0 は 255 にクリップされる。
        assert_eq!(plane.get(4, 4), 255);
    }
}
