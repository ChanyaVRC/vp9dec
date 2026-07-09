//! デブロッキング（ループ）フィルタ（仕様 8.8 節 "Loop filter process"）。
//!
//! 仕様 8.8 節に忠実に、フレーム全体を次の入れ子ループで走査する（節冒頭のフレーム全体の
//! 走査擬似コードそのまま）:
//!
//! ```text
//! for ( row = 0; row < MiRows; row += 8 )
//!   for ( col = 0; col < MiCols; col += 8 )
//!     for ( plane = 0; plane < 3; plane++ )
//!       for ( pass = 0; pass < 2; pass++ )
//!         superblock loop filter process( plane, pass, row, col )
//! ```
//!
//! `pass == 0` が縦エッジ（左右のブロック境界）、`pass == 1` が横エッジ（上下のブロック境界）
//! を意味する。同じサンプルが複数回フィルタされ得るため、この走査順（縦→横、スーパーブロック
//! ラスタ順）を厳密に守る必要がある（仕様 8.8 節の NOTE を参照）。
//!
//! # 既知の簡略化
//!
//! - `isIntra`（`RefFrames[row][col][0] <= INTRA_FRAME`）・`modeType`（`YModes` が
//!   `NEARESTMV`/`NEARMV`/`NEWMV` かどうか）は M3 で `MiInfo` に追加した `ref_frame`/`y_mode`
//!   から実値を参照する（`superblock_loop_filter` 参照）。
//! - `segmentation_enabled == true` のフレームは `tile.rs` が `TileError::SegmentationNotSupported`
//!   として拒否するため、`seg_feature_active( SEG_LVL_ALT_L )` は常に偽と仮定できる
//!   （仕様 8.8.1 節 手順 2 は発生しない）。
//! - `loop_filter_ref_deltas`/`loop_filter_mode_deltas` はフレーム間で持続する状態だが
//!   （仕様 7.2 節 `setup_past_independence`）、本実装は毎フレーム `parse_loop_filter_params`
//!   内でデフォルト値 `[1, 0, -1, -1]`/`[0, 0]` から起動する（前フレームからの引き継ぎ未実装）。
//!   これはループフィルタの出力画素にのみ影響し、ビットストリームの読み取りには影響しない
//!   （`loop_filter_params()` が読むビット数は現在のデルタ値に依存しないため）。M3 後半で
//!   フレーム間状態として引き継ぐよう改修する。

use crate::framebuffer::Plane;
use crate::header::LoopFilterParams;
use crate::prob_tables::{
    BLOCK_16X16, BLOCK_8X8, MAX_TXSIZE_LOOKUP, NEARESTMV, NEARMV, NEWMV,
    NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP, SS_SIZE_LOOKUP, TX_16X16, TX_4X4,
    TX_8X8,
};
use crate::tile::MiGrid;

const MAX_SEGMENTS: usize = 8;
const MAX_REF_FRAMES: usize = 4;
const MAX_MODE_LF_DELTAS: usize = 2;
const MAX_LOOP_FILTER: i32 = 63;
/// 参照フレーム種別のインデックス（仕様の `RefFrame` 列挙のうち本実装が使う値のみ）。
const INTRA_FRAME: usize = 0;

/// `LvlLookup[ segmentId ][ ref ][ mode ]`（仕様 8.8.1 節）。
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

/// `get_uv_tx_size()`（仕様 6.4.22 節）。`src/tile.rs` の `TileDecoder::get_uv_tx_size` と
/// 等価だが、ループフィルタは `TileDecoder` の外（`&self` を持たない自由関数）で完結させる
/// ため、ここに小さく複製している。
fn get_uv_tx_size(mi_size: u8, tx_size: u8, subsampling_x: u32, subsampling_y: u32) -> u8 {
    if mi_size < BLOCK_8X8 {
        return TX_4X4;
    }
    let plane_sz = SS_SIZE_LOOKUP[mi_size as usize][subsampling_x as usize][subsampling_y as usize];
    tx_size.min(MAX_TXSIZE_LOOKUP[plane_sz as usize])
}

/// 仕様 8.8.1 節 "Loop filter frame init process"。
fn build_lvl_lookup(lf: &LoopFilterParams) -> LvlLookup {
    // nShift はフレームレベルの loop_filter_level から一度だけ計算される
    // （lvlSeg からではない点に注意。仕様本文どおり）。
    let n_shift = (lf.level as i32) >> 5;
    let mut table: LvlLookup = [[[0; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]; MAX_SEGMENTS];

    for seg in table.iter_mut() {
        // segmentation_enabled == true のフレームは呼び出し前に拒否されているため、
        // seg_feature_active(SEG_LVL_ALT_L) は常に偽（手順 2 は適用しない）。
        let lvl_seg = lf.level as i32;

        if !lf.delta_enabled {
            for r in seg.iter_mut() {
                for m in r.iter_mut() {
                    *m = lvl_seg;
                }
            }
        } else {
            let intra_lvl = lvl_seg + (lf.ref_deltas[INTRA_FRAME] as i32) * (1 << n_shift);
            seg[INTRA_FRAME][0] = clip3(0, MAX_LOOP_FILTER, intra_lvl);
            // seg[INTRA_FRAME][1] は仕様上定義されない（INTRA_FRAME 行は mode=0 のみ）。
            // キーフレームでは isIntra が常に真で modeType が常に 0 なので参照されない。
            for (r, ref_delta) in lf.ref_deltas.iter().enumerate().skip(1) {
                for (m, mode_delta) in lf.mode_deltas.iter().enumerate() {
                    let inter_lvl = lvl_seg
                        + (*ref_delta as i32) * (1 << n_shift)
                        + (*mode_delta as i32) * (1 << n_shift);
                    seg[r][m] = clip3(0, MAX_LOOP_FILTER, inter_lvl);
                }
            }
        }
    }
    table
}

/// 仕様 8.8.4 節 "Adaptive filter strength process" のうち `limit`/`blimit`/`thresh` の計算。
/// （`lvl` は呼び出し側で `LvlLookup` から求めて渡す。）
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

/// 仕様 8.8.3 節 "Filter size process"。
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

/// `x + dx*k, y + dy*k` の位置のサンプルを読む（仕様の `q_k`/`p_k` の一般形。
/// `k` が負のとき `p` 側、非負のとき `q` 側を表す）。
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

/// 仕様 8.8.5.1 節 "Filter mask process"。戻り値は `(hevMask, filterMask, flatMask, flatMask2)`。
/// `BitDepth == 8` 固定（本デコーダは 8bit のみ対応、`decode_keyframe` 参照）のため、
/// 仕様の `<< (BitDepth - 8)` によるビット深度スケーリングは恒等（シフト量 0）として省略した。
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

/// 仕様 8.8.5.2 節 "Narrow filter process"（`filter4`）。`BitDepth == 8` 固定。
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

/// 仕様 8.8.5.3 節 "Wide filter process"。`log2_size` は 3（8タップ）または 4（16タップ）。
fn wide_filter(plane: &mut Plane, x: usize, y: usize, dx: i64, dy: i64, log2_size: u32) {
    let n: i64 = (1i64 << (log2_size - 1)) - 1;
    // F のインデックスは -n..n-1。オフセット n を足して 0 始まりの配列に格納する。
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

/// 仕様 8.8.5 節 "Sample filtering process"。
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

/// 仕様 8.8.2 節 "Superblock loop filter process"。
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
            // 仕様 8.8.2 節手順9: isIntra = RefFrames[loopRow][loopCol][0] <= INTRA_FRAME。
            let ref_frame = mi.ref_frame[0];
            let is_intra = ref_frame == INTRA_FRAME as u8;

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

            // 仕様 8.8.4 節: modeType = 1 if mode in {NEARESTMV,NEARMV,NEWMV} else 0。
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

/// 仕様 8.8 節 "Loop filter process" のフレーム全体エントリポイント。
///
/// `planes` は `CurrFrame`（`TileDecoder::planes`/`planes_mut` 相当、スーパーブロック境界まで
/// 確保済みのバッファ）、`mi_grid` はモード情報グリッド、`mi_cols`/`mi_rows` は
/// `compute_image_size()` で得られる非パディングの mi 単位フレームサイズ。
pub fn loop_filter_frame(
    planes: &mut [Plane; 3],
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lf: &LoopFilterParams,
) {
    let lvl_lookup = build_lvl_lookup(lf);

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

    #[test]
    fn lvl_lookup_without_deltas_is_flat_level() {
        let lf = LoopFilterParams {
            level: 20,
            sharpness: 0,
            delta_enabled: false,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        };
        let table = build_lvl_lookup(&lf);
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
        // level=40 -> nShift = 40>>5 = 1。ref_deltas[INTRA_FRAME] のデフォルトは 1。
        let lf = LoopFilterParams {
            level: 40,
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        };
        let table = build_lvl_lookup(&lf);
        // intraLvl = 40 + 1*(1<<1) = 42
        assert_eq!(table[0][INTRA_FRAME][0], 42);
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
        // 完全に平坦な入力（すべて同じ値）はフィルタしても変化しないはず。
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
