//! 動きベクトル予測で使う純粋な補助関数（仕様 6.5 節 "Motion vector prediction"）。
//!
//! `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs` 本体は `MiGrid`（フレーム内の
//! モード情報グリッド）や `TileDecoder` の状態（タイル境界・フレームサイズ等）に強く依存する
//! ため `src/tile.rs` に実装するが、それらが呼び出す純粋な計算（クランプ・符号反転・
//! しきい値判定）はここに独立させ、単体テストしやすくしている。

/// 動きベクトル（`[row, col]`、単位は 1/8 pel）。
pub type Mv = [i32; 2];

/// `ZeroMv`（仕様 6.4.18 節ほか）。
pub const ZERO_MV: Mv = [0, 0];

/// `MVREF_NEIGHBOURS`（仕様 3 節、定数一覧）。`find_mv_refs` が走査する近傍候補数。
pub const MVREF_NEIGHBOURS: usize = 8;

/// `MV_BORDER`（仕様 3 節）。`find_mv_refs` 末尾の `clamp_mv_ref` で使うクランプ境界。
pub const MV_BORDER: i32 = 128;

/// `(BORDERINPIXELS - INTERP_EXTEND) << 3`（仕様 6.5.12 節）。`find_best_ref_mvs` で使う
/// クランプ境界（`BORDERINPIXELS = 160`, `INTERP_EXTEND = 4`）。
pub const MV_PRED_BORDER: i32 = (160 - 4) << 3;

/// `COMPANDED_MVREF_THRESH`（仕様 3 節）。`use_mv_hp` の判定しきい値。
const COMPANDED_MVREF_THRESH: i32 = 8;

/// `MI_SIZE`（仕様 3 節）。8x8 mode info 単位の一辺の画素数。
const MI_SIZE: i32 = 8;

/// `clamp_mv_row( mvec, border )`（仕様 6.5.4 節）。
pub fn clamp_mv_row(mvec: i32, border: i32, mi_row: u32, bh: u32, mi_rows: u32) -> i32 {
    let mb_to_top_edge = -((mi_row as i32) * MI_SIZE * 8);
    let mb_to_bottom_edge = ((mi_rows as i32) - (bh as i32) - (mi_row as i32)) * MI_SIZE * 8;
    mvec.clamp(mb_to_top_edge - border, mb_to_bottom_edge + border)
}

/// `clamp_mv_col( mvec, border )`（仕様 6.5.5 節）。
pub fn clamp_mv_col(mvec: i32, border: i32, mi_col: u32, bw: u32, mi_cols: u32) -> i32 {
    let mb_to_left_edge = -((mi_col as i32) * MI_SIZE * 8);
    let mb_to_right_edge = ((mi_cols as i32) - (bw as i32) - (mi_col as i32)) * MI_SIZE * 8;
    mvec.clamp(mb_to_left_edge - border, mb_to_right_edge + border)
}

/// `add_mv_ref_list( refList )`（仕様 6.5.6 節）。`RefListMv`/`RefMvCount` への追加処理。
/// 重複除去（`RefListMv[0]` と同じ値は追加しない）と、最大 2 件までの上限を扱う。
pub fn add_mv_ref_list(ref_list_mv: &mut [Mv; 2], ref_mv_count: &mut usize, candidate: Mv) {
    if *ref_mv_count >= 2 {
        return;
    }
    if *ref_mv_count > 0 && candidate == ref_list_mv[0] {
        return;
    }
    ref_list_mv[*ref_mv_count] = candidate;
    *ref_mv_count += 1;
}

/// `scale_mv( refList, refFrame )`（仕様 6.5.9 節）。候補フレームと目的の参照フレームで
/// `ref_frame_sign_bias` が異なる場合、動きベクトルの符号を反転する。
pub fn scale_mv(
    candidate_mv: Mv,
    cand_frame: u8,
    ref_frame: u8,
    ref_frame_sign_bias: &[bool; 4],
) -> Mv {
    if ref_frame_sign_bias[cand_frame as usize] != ref_frame_sign_bias[ref_frame as usize] {
        [-candidate_mv[0], -candidate_mv[1]]
    } else {
        candidate_mv
    }
}

/// `use_mv_hp( deltaMv )`（仕様 6.5.13 節）。
pub fn use_mv_hp(delta_mv: Mv) -> bool {
    (delta_mv[0].abs() >> 3) < COMPANDED_MVREF_THRESH
        && (delta_mv[1].abs() >> 3) < COMPANDED_MVREF_THRESH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_mv_row_matches_spec_formula() {
        // MiRow=0, bh=1, MiRows=8 (64px フレーム) の 8x8 ブロック。
        // mbToTopEdge = 0, mbToBottomEdge = (8-1-0)*8*8 = 448
        assert_eq!(clamp_mv_row(1000, 128, 0, 1, 8), 448 + 128);
        assert_eq!(clamp_mv_row(-1000, 128, 0, 1, 8), 0 - 128);
        assert_eq!(clamp_mv_row(10, 128, 0, 1, 8), 10);
    }

    #[test]
    fn add_mv_ref_list_deduplicates_and_caps_at_two() {
        let mut list = [ZERO_MV; 2];
        let mut count = 0usize;
        add_mv_ref_list(&mut list, &mut count, [1, 2]);
        assert_eq!(count, 1);
        add_mv_ref_list(&mut list, &mut count, [1, 2]); // 重複 -> 追加されない
        assert_eq!(count, 1);
        add_mv_ref_list(&mut list, &mut count, [3, 4]);
        assert_eq!(count, 2);
        assert_eq!(list, [[1, 2], [3, 4]]);
        add_mv_ref_list(&mut list, &mut count, [5, 6]); // 上限 2 で無視
        assert_eq!(count, 2);
        assert_eq!(list, [[1, 2], [3, 4]]);
    }

    #[test]
    fn scale_mv_flips_sign_on_differing_bias() {
        let sign_bias = [false, false, true, false];
        assert_eq!(scale_mv([4, -2], 1, 2, &sign_bias), [-4, 2]);
        assert_eq!(scale_mv([4, -2], 1, 3, &sign_bias), [4, -2]);
    }

    #[test]
    fn use_mv_hp_threshold() {
        assert!(use_mv_hp([63, 0])); // 63>>3 = 7 < 8
        assert!(!use_mv_hp([64, 0])); // 64>>3 = 8, not < 8
    }
}
