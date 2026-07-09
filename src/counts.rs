//! 確率適応（backward adaptation, 仕様 8.4 節）のためのカウンタ収集・`merge_prob`/
//! `merge_probs`・`adapt_coef_probs`/`adapt_noncoef_probs`。
//!
//! カウンタの形状・更新条件は仕様 8.3 節（`clear_counts`、各配列の次元）と
//! 9.3.4 節（"Syntax element counting process"、どのシンタックス要素がどのカウンタを
//! 更新するかの対応表）に厳密に従う。`src/tile.rs` の各シンタックス読み取り箇所が
//! この構造体のフィールドをインクリメントする。

use crate::compressed_header::CompressedHeaderProbs;
use crate::prob_tables::{
    CoefProbs, BINARY_TREE, COUNT_SAT, INTERP_FILTER_TREE, INTER_MODE_TREE, INTRA_MODE_TREE,
    MAX_UPDATE_FACTOR, MV_CLASS_TREE, MV_FR_TREE, MV_JOINT_TREE, PARTITION_TREE, SMALL_TOKEN_TREE,
    SWITCHABLE, TX_16X16, TX_32X32, TX_8X8, TX_MODE_SELECT,
};

/// `counts_token[TX_SIZES][BLOCK_TYPES][REF_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS][UNCONSTRAINED_NODES]`。
pub type TokenCounts = [[[[[[u32; 3]; 6]; 6]; 2]; 2]; 4];
/// `counts_more_coefs[TX_SIZES][BLOCK_TYPES][REF_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS][2]`。
pub type MoreCoefsCounts = [[[[[[u32; 2]; 6]; 6]; 2]; 2]; 4];

/// 仕様 8.3 節 "Clear counts process" が列挙するすべてのカウンタ配列。
#[derive(Debug, Clone)]
pub struct Counts {
    pub intra_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub interp_filter: [[u32; 3]; 4],
    pub inter_mode: [[u32; 4]; 7],
    /// `counts_tx_size[TX_SIZES][TX_SIZE_CONTEXTS][TX_SIZES]`。`tx_size[maxTxSize][ctx][value]`。
    pub tx_size: [[[u32; 4]; 2]; 4],
    pub is_inter: [[u32; 2]; 4],
    pub comp_mode: [[u32; 2]; 5],
    /// `counts_single_ref[REF_CONTEXTS][2][2]`。`single_ref[ctx][p1_or_p2][value]`。
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub skip: [[u32; 2]; 3],
    pub mv_joint: [u32; 4],
    pub mv_sign: [[u32; 2]; 2],
    pub mv_class: [[u32; 11]; 2],
    pub mv_class0_bit: [[u32; 2]; 2],
    pub mv_class0_fr: [[[u32; 4]; 2]; 2],
    pub mv_class0_hp: [[u32; 2]; 2],
    pub mv_bits: [[[u32; 2]; 10]; 2],
    pub mv_fr: [[u32; 4]; 2],
    pub mv_hp: [[u32; 2]; 2],
    pub token: TokenCounts,
    pub more_coefs: MoreCoefsCounts,
}

impl Default for Counts {
    fn default() -> Self {
        Self::new()
    }
}

impl Counts {
    pub fn new() -> Self {
        Self {
            intra_mode: [[0; 10]; 4],
            uv_mode: [[0; 10]; 10],
            partition: [[0; 4]; 16],
            interp_filter: [[0; 3]; 4],
            inter_mode: [[0; 4]; 7],
            tx_size: [[[0; 4]; 2]; 4],
            is_inter: [[0; 2]; 4],
            comp_mode: [[0; 2]; 5],
            single_ref: [[[0; 2]; 2]; 5],
            comp_ref: [[0; 2]; 5],
            skip: [[0; 2]; 3],
            mv_joint: [0; 4],
            mv_sign: [[0; 2]; 2],
            mv_class: [[0; 11]; 2],
            mv_class0_bit: [[0; 2]; 2],
            mv_class0_fr: [[[0; 4]; 2]; 2],
            mv_class0_hp: [[0; 2]; 2],
            mv_bits: [[[0; 2]; 10]; 2],
            mv_fr: [[0; 4]; 2],
            mv_hp: [[0; 2]; 2],
            token: [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4],
            more_coefs: [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4],
        }
    }
}

/// `merge_prob( preProb, ct0, ct1, countSat, maxUpdateFactor )`（仕様 8.4.1 節）。
fn merge_prob(pre_prob: u8, ct0: u32, ct1: u32, count_sat: u32, max_update_factor: u32) -> u8 {
    let den = ct0 + ct1;
    let prob = if den == 0 {
        128
    } else {
        (((ct0 as u64 * 256 + (den as u64 >> 1)) / den as u64) as i64).clamp(1, 255) as u8
    };
    let count = den.min(count_sat);
    let factor = max_update_factor * count / count_sat;
    let out = (pre_prob as i64 * (256 - factor as i64) + prob as i64 * factor as i64 + 128) >> 8;
    out as u8
}

/// `merge_probs( tree, i, probs, counts, countSat, maxUpdateFactor )`（仕様 8.4.2 節）。
/// 戻り値は `leftCount + rightCount`。
fn merge_probs(
    tree: &[i32],
    i: usize,
    probs: &mut [u8],
    counts: &[u32],
    count_sat: u32,
    max_update_factor: u32,
) -> u32 {
    let s = tree[i];
    let left_count = if s <= 0 {
        counts[(-s) as usize]
    } else {
        merge_probs(
            tree,
            s as usize,
            probs,
            counts,
            count_sat,
            max_update_factor,
        )
    };
    let r = tree[i + 1];
    let right_count = if r <= 0 {
        counts[(-r) as usize]
    } else {
        merge_probs(
            tree,
            r as usize,
            probs,
            counts,
            count_sat,
            max_update_factor,
        )
    };
    probs[i >> 1] = merge_prob(
        probs[i >> 1],
        left_count,
        right_count,
        count_sat,
        max_update_factor,
    );
    left_count + right_count
}

/// `adapt_probs( tree, probs, counts )`（仕様 8.4.4 節）。
fn adapt_probs(tree: &[i32], probs: &mut [u8], counts: &[u32]) {
    merge_probs(tree, 0, probs, counts, COUNT_SAT, MAX_UPDATE_FACTOR);
}

/// `adapt_prob( prob, counts )`（仕様 8.4.4 節）。
fn adapt_prob(prob: u8, counts: [u32; 2]) -> u8 {
    merge_prob(prob, counts[0], counts[1], COUNT_SAT, MAX_UPDATE_FACTOR)
}

/// `adapt_coef_probs( )`（仕様 8.4.3 節）。`update_factor` は呼び出し側
/// （`FrameIsIntra`/`LastFrameType` に基づく、仕様本文参照）が決定して渡す。
///
/// `t`/`i`/`j`/`k`/`l` は `coef_probs`/`counts.token`/`counts.more_coefs` の 3 つを同時に
/// 同じ添字で辿るため、`iter_mut().enumerate()` に書き換えると却って読みにくくなる
/// （仕様の擬似コードの `for` ループ構造をそのまま Rust の添字ループへ転記したもの）。
#[allow(clippy::needless_range_loop)]
pub fn adapt_coef_probs(coef_probs: &mut CoefProbs, counts: &Counts, update_factor: u32) {
    for t in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for l in 0..max_l {
                        let probs = &mut coef_probs[t][i][j][k][l];
                        merge_probs(
                            &SMALL_TOKEN_TREE,
                            2,
                            probs,
                            &counts.token[t][i][j][k][l],
                            24,
                            update_factor,
                        );
                        merge_probs(
                            &BINARY_TREE,
                            0,
                            probs,
                            &counts.more_coefs[t][i][j][k][l],
                            24,
                            update_factor,
                        );
                    }
                }
            }
        }
    }
}

/// `adapt_noncoef_probs( )`（仕様 8.4.4 節）。
pub fn adapt_noncoef_probs(
    probs: &mut CompressedHeaderProbs,
    counts: &Counts,
    interpolation_filter: u8,
    tx_mode: u8,
    allow_high_precision_mv: bool,
) {
    for i in 0..4 {
        probs.is_inter_prob[i] = adapt_prob(probs.is_inter_prob[i], counts.is_inter[i]);
    }
    for i in 0..5 {
        probs.comp_mode_prob[i] = adapt_prob(probs.comp_mode_prob[i], counts.comp_mode[i]);
    }
    for i in 0..5 {
        probs.comp_ref_prob[i] = adapt_prob(probs.comp_ref_prob[i], counts.comp_ref[i]);
    }
    for i in 0..5 {
        for j in 0..2 {
            probs.single_ref_prob[i][j] =
                adapt_prob(probs.single_ref_prob[i][j], counts.single_ref[i][j]);
        }
    }
    for i in 0..7 {
        adapt_probs(
            &INTER_MODE_TREE,
            &mut probs.inter_mode_probs[i],
            &counts.inter_mode[i],
        );
    }
    for i in 0..4 {
        adapt_probs(
            &INTRA_MODE_TREE,
            &mut probs.y_mode_probs[i],
            &counts.intra_mode[i],
        );
    }
    for i in 0..10 {
        adapt_probs(
            &INTRA_MODE_TREE,
            &mut probs.uv_mode_probs[i],
            &counts.uv_mode[i],
        );
    }
    for i in 0..16 {
        adapt_probs(
            &PARTITION_TREE,
            &mut probs.partition_probs[i],
            &counts.partition[i],
        );
    }
    for i in 0..3 {
        probs.skip_prob[i] = adapt_prob(probs.skip_prob[i], counts.skip[i]);
    }
    if interpolation_filter == SWITCHABLE {
        for i in 0..4 {
            adapt_probs(
                &INTERP_FILTER_TREE,
                &mut probs.interp_filter_probs[i],
                &counts.interp_filter[i],
            );
        }
    }
    if tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            adapt_probs(
                &crate::prob_tables::TX_SIZE_8_TREE,
                &mut probs.tx_probs[TX_8X8 as usize][i],
                &counts.tx_size[TX_8X8 as usize][i],
            );
            adapt_probs(
                &crate::prob_tables::TX_SIZE_16_TREE,
                &mut probs.tx_probs[TX_16X16 as usize][i],
                &counts.tx_size[TX_16X16 as usize][i],
            );
            adapt_probs(
                &crate::prob_tables::TX_SIZE_32_TREE,
                &mut probs.tx_probs[TX_32X32 as usize][i],
                &counts.tx_size[TX_32X32 as usize][i],
            );
        }
    }
    adapt_probs(&MV_JOINT_TREE, &mut probs.mv_joint_probs, &counts.mv_joint);
    for i in 0..2 {
        probs.mv_sign_prob[i] = adapt_prob(probs.mv_sign_prob[i], counts.mv_sign[i]);
        adapt_probs(
            &MV_CLASS_TREE,
            &mut probs.mv_class_probs[i],
            &counts.mv_class[i],
        );
        probs.mv_class0_bit_prob[i] =
            adapt_prob(probs.mv_class0_bit_prob[i], counts.mv_class0_bit[i]);
        for j in 0..10 {
            probs.mv_bits_prob[i][j] = adapt_prob(probs.mv_bits_prob[i][j], counts.mv_bits[i][j]);
        }
        for j in 0..2 {
            adapt_probs(
                &MV_FR_TREE,
                &mut probs.mv_class0_fr_probs[i][j],
                &counts.mv_class0_fr[i][j],
            );
        }
        adapt_probs(&MV_FR_TREE, &mut probs.mv_fr_probs[i], &counts.mv_fr[i]);
        if allow_high_precision_mv {
            probs.mv_class0_hp_prob[i] =
                adapt_prob(probs.mv_class0_hp_prob[i], counts.mv_class0_hp[i]);
            probs.mv_hp_prob[i] = adapt_prob(probs.mv_hp_prob[i], counts.mv_hp[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_prob_with_no_observations_keeps_preprob_untouched_direction() {
        // den == 0 のとき prob=128、count=0 なので factor=0、out=preProb がそのまま返る。
        assert_eq!(merge_prob(200, 0, 0, 20, 128), 200);
    }

    #[test]
    fn merge_prob_saturates_toward_observed_ratio_with_enough_counts() {
        // ct0=0, ct1=100 (count_sat=20 に達している) -> factor=maxUpdateFactor=128 のとき
        // outProb = Round2( preProb*128 + 1*128, 8 ) 程度に preProb から離れて 1 に近づく。
        let out = merge_prob(200, 0, 100, 20, 128);
        assert!(out < 200);
    }

    #[test]
    fn merge_probs_binary_tree_updates_single_prob() {
        let mut probs = [100u8, 0, 0];
        let counts = [10u32, 30u32];
        merge_probs(&BINARY_TREE, 0, &mut probs, &counts, 20, 128);
        // ct1 (=30) の方が多いので prob は下がる方向に動く。
        assert!(probs[0] < 100);
    }
}
