//! Counter collection and `merge_prob`/`merge_probs`/`adapt_coef_probs`/
//! `adapt_noncoef_probs` for probability adaptation (backward adaptation, spec §8.4).
//!
//! The shape and update conditions of the counters strictly follow spec §8.3
//! (`clear_counts`, each array's dimensions) and §9.3.4 ("Syntax element
//! counting process", the table mapping which syntax element updates which
//! counter). Each syntax-element read site in `src/tile.rs` increments the
//! fields of this struct.

use crate::compressed_header::CompressedHeaderProbs;
use crate::prob_tables::{
    CoefProbs, BINARY_TREE, COUNT_SAT, INTERP_FILTER_TREE, INTER_MODE_TREE, INTRA_MODE_TREE,
    MAX_UPDATE_FACTOR, MV_CLASS_TREE, MV_FR_TREE, MV_JOINT_TREE, PARTITION_TREE, SMALL_TOKEN_TREE,
    SWITCHABLE, TX_16X16, TX_32X32, TX_8X8, TX_MODE_SELECT,
};

/// `counts_token[TX_SIZES][BLOCK_TYPES][REF_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS][UNCONSTRAINED_NODES]`.
pub type TokenCounts = [[[[[[u32; 3]; 6]; 6]; 2]; 2]; 4];
/// `counts_more_coefs[TX_SIZES][BLOCK_TYPES][REF_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS][2]`.
pub type MoreCoefsCounts = [[[[[[u32; 2]; 6]; 6]; 2]; 2]; 4];

/// All counter arrays enumerated by the "Clear counts process" in spec §8.3.
///
/// `#[repr(C)]` + the `const` size assertion below make [`Counts::add_assign`]'s flat-`u32`
/// reinterpretation sound by construction (guaranteed field-order layout, provably no
/// padding), not by layout luck.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Counts {
    pub intra_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub interp_filter: [[u32; 3]; 4],
    pub inter_mode: [[u32; 4]; 7],
    /// `counts_tx_size[TX_SIZES][TX_SIZE_CONTEXTS][TX_SIZES]`. `tx_size[maxTxSize][ctx][value]`.
    pub tx_size: [[[u32; 4]; 2]; 4],
    pub is_inter: [[u32; 2]; 4],
    pub comp_mode: [[u32; 2]; 5],
    /// `counts_single_ref[REF_CONTEXTS][2][2]`. `single_ref[ctx][p1_or_p2][value]`.
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

/// Compile-time no-padding proof for [`Counts::add_assign`]: with `#[repr(C)]` the struct is
/// laid out field-by-field in declaration order, so its size equals the summed field sizes
/// exactly when there is no padding anywhere. Adding/reshaping a field breaks this assertion
/// until the sum (and `add_assign`'s flat view) is re-audited.
const _: () = assert!(
    std::mem::size_of::<Counts>()
        == std::mem::size_of::<[[u32; 10]; 4]>()      // intra_mode
            + std::mem::size_of::<[[u32; 10]; 10]>()  // uv_mode
            + std::mem::size_of::<[[u32; 4]; 16]>()   // partition
            + std::mem::size_of::<[[u32; 3]; 4]>()    // interp_filter
            + std::mem::size_of::<[[u32; 4]; 7]>()    // inter_mode
            + std::mem::size_of::<[[[u32; 4]; 2]; 4]>() // tx_size
            + std::mem::size_of::<[[u32; 2]; 4]>()    // is_inter
            + std::mem::size_of::<[[u32; 2]; 5]>()    // comp_mode
            + std::mem::size_of::<[[[u32; 2]; 2]; 5]>() // single_ref
            + std::mem::size_of::<[[u32; 2]; 5]>()    // comp_ref
            + std::mem::size_of::<[[u32; 2]; 3]>()    // skip
            + std::mem::size_of::<[u32; 4]>()         // mv_joint
            + std::mem::size_of::<[[u32; 2]; 2]>()    // mv_sign
            + std::mem::size_of::<[[u32; 11]; 2]>()   // mv_class
            + std::mem::size_of::<[[u32; 2]; 2]>()    // mv_class0_bit
            + std::mem::size_of::<[[[u32; 4]; 2]; 2]>() // mv_class0_fr
            + std::mem::size_of::<[[u32; 2]; 2]>()    // mv_class0_hp
            + std::mem::size_of::<[[[u32; 2]; 10]; 2]>() // mv_bits
            + std::mem::size_of::<[[u32; 4]; 2]>()    // mv_fr
            + std::mem::size_of::<[[u32; 2]; 2]>()    // mv_hp
            + std::mem::size_of::<TokenCounts>()      // token
            + std::mem::size_of::<MoreCoefsCounts>() // more_coefs
);

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

    /// Adds every counter in `other` into `self`, element-wise. Used to merge the per-tile-column
    /// `Counts` that tile-parallel decode accumulates on separate threads back into one frame
    /// total before backward probability adaptation. Order-independent (integer addition is
    /// associative/commutative), so the merged total is identical to a single-threaded decode's.
    ///
    /// Every field of `Counts` is a `u32` array, so the struct is a contiguous block of `u32`
    /// with no padding; summing the flat `u32` view is exactly a field-wise sum but avoids
    /// enumerating all 28 fields. `#[repr(C)]` + the `const` size assertion at the struct
    /// prove the no-padding layout at compile time; the sibling unit test additionally checks
    /// the summing behavior across field shapes.
    pub fn add_assign(&mut self, other: &Counts) {
        debug_assert_eq!(std::mem::size_of::<Counts>() % 4, 0);
        let n = std::mem::size_of::<Counts>() / 4;
        // SAFETY: `Counts` is `#[repr(C)]` with only `u32`-array fields and provably no
        // padding (the `const` assertion above), so it is soundly viewed as `n` contiguous
        // `u32`s; `self` and `other` are the same type, so their flat views line up
        // field-for-field.
        let dst = unsafe { std::slice::from_raw_parts_mut(self as *mut Counts as *mut u32, n) };
        let src = unsafe { std::slice::from_raw_parts(other as *const Counts as *const u32, n) };
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = d.wrapping_add(s);
        }
    }
}

/// `merge_prob( preProb, ct0, ct1, countSat, maxUpdateFactor )` (spec §8.4.1).
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

/// `merge_probs( tree, i, probs, counts, countSat, maxUpdateFactor )` (spec §8.4.2).
/// Returns `leftCount + rightCount`.
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

/// `adapt_probs( tree, probs, counts )` (spec §8.4.4).
fn adapt_probs(tree: &[i32], probs: &mut [u8], counts: &[u32]) {
    merge_probs(tree, 0, probs, counts, COUNT_SAT, MAX_UPDATE_FACTOR);
}

/// `adapt_prob( prob, counts )` (spec §8.4.4).
fn adapt_prob(prob: u8, counts: [u32; 2]) -> u8 {
    merge_prob(prob, counts[0], counts[1], COUNT_SAT, MAX_UPDATE_FACTOR)
}

/// `adapt_coef_probs( )` (spec §8.4.3). `update_factor` is determined by the
/// caller (based on `FrameIsIntra`/`LastFrameType`; see the spec text) and passed in.
///
/// `t`/`i`/`j`/`k`/`l` walk `coef_probs`/`counts.token`/`counts.more_coefs`
/// simultaneously with the same indices, so rewriting this with
/// `iter_mut().enumerate()` would only make it harder to read (this is a
/// direct transcription of the spec's pseudocode `for`-loop structure into a
/// Rust indexed loop).
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

/// `adapt_noncoef_probs( )` (spec §8.4.4).
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
    fn add_assign_sums_counters_across_all_field_shapes() {
        // Guards the flat-u32-reinterpret layout assumption in `Counts::add_assign`: set fields of
        // several different shapes (2D/1D/6D nested arrays, first/middle/last struct fields) and
        // confirm each sums independently, and that an untouched field stays 0.
        let mut a = Counts::new();
        let mut b = Counts::new();
        a.intra_mode[1][2] = 5; // first field
        a.mv_joint[3] = 4;
        a.token[3][1][0][5][4][2] = 7; // second-to-last field
        b.intra_mode[1][2] = 10;
        b.mv_joint[3] = 100;
        b.token[3][1][0][5][4][2] = 1;
        b.more_coefs[2][1][1][3][2][0] = 9; // last field
        a.add_assign(&b);
        assert_eq!(a.intra_mode[1][2], 15);
        assert_eq!(a.mv_joint[3], 104);
        assert_eq!(a.token[3][1][0][5][4][2], 8);
        assert_eq!(a.more_coefs[2][1][1][3][2][0], 9);
        assert_eq!(a.uv_mode[0][0], 0, "untouched field must stay 0");
    }

    #[test]
    fn merge_prob_with_no_observations_keeps_preprob_untouched_direction() {
        // When den == 0, prob=128 and count=0, so factor=0, and out=preProb is returned as-is.
        assert_eq!(merge_prob(200, 0, 0, 20, 128), 200);
    }

    #[test]
    fn merge_prob_saturates_toward_observed_ratio_with_enough_counts() {
        // ct0=0, ct1=100 (reaches count_sat=20) -> when factor=maxUpdateFactor=128,
        // outProb moves away from preProb toward 1, roughly Round2( preProb*128 + 1*128, 8 ).
        let out = merge_prob(200, 0, 100, 20, 128);
        assert!(out < 200);
    }

    #[test]
    fn merge_probs_binary_tree_updates_single_prob() {
        let mut probs = [100u8, 0, 0];
        let counts = [10u32, 30u32];
        merge_probs(&BINARY_TREE, 0, &mut probs, &counts, 20, 128);
        // Since ct1 (=30) is larger, prob moves in the decreasing direction.
        assert!(probs[0] < 100);
    }
}
