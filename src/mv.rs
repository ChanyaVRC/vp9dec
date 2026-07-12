//! Pure helper functions used by motion vector prediction (spec §6.5 "Motion vector prediction").
//!
//! The bodies of `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs` are
//! implemented in `src/tile.rs` because they depend heavily on `MiGrid` (the
//! in-frame mode info grid) and `TileDecoder` state (tile boundaries, frame
//! size, etc.), but the pure computations they call (clamping, sign
//! inversion, threshold checks) are factored out here on their own to make
//! them easy to unit test.

/// A motion vector (`[row, col]`, in units of 1/8 pel).
pub type Mv = [i32; 2];

/// `ZeroMv` (spec §6.4.18 and others).
pub const ZERO_MV: Mv = [0, 0];

/// `MVREF_NEIGHBOURS` (spec §3, constants list). The number of neighbor
/// candidates `find_mv_refs` scans.
pub const MVREF_NEIGHBOURS: usize = 8;

/// `MV_BORDER` (spec §3). The clamp border used by `clamp_mv_ref` at the end
/// of `find_mv_refs`.
pub const MV_BORDER: i32 = 128;

/// `(BORDERINPIXELS - INTERP_EXTEND) << 3` (spec §6.5.12). The clamp border
/// used by `find_best_ref_mvs` (`BORDERINPIXELS = 160`, `INTERP_EXTEND = 4`).
pub const MV_PRED_BORDER: i32 = (160 - 4) << 3;

/// `COMPANDED_MVREF_THRESH` (spec §3). The threshold used by `use_mv_hp`.
const COMPANDED_MVREF_THRESH: i32 = 8;

/// `MI_SIZE` (spec §3). The side length in pixels of an 8x8 mode info unit.
const MI_SIZE: i32 = 8;

/// `clamp_mv_row( mvec, border )` (spec §6.5.4).
pub fn clamp_mv_row(mvec: i32, border: i32, mi_row: u32, bh: u32, mi_rows: u32) -> i32 {
    let mb_to_top_edge = -((mi_row as i32) * MI_SIZE * 8);
    let mb_to_bottom_edge = ((mi_rows as i32) - (bh as i32) - (mi_row as i32)) * MI_SIZE * 8;
    mvec.clamp(mb_to_top_edge - border, mb_to_bottom_edge + border)
}

/// `clamp_mv_col( mvec, border )` (spec §6.5.5).
pub fn clamp_mv_col(mvec: i32, border: i32, mi_col: u32, bw: u32, mi_cols: u32) -> i32 {
    let mb_to_left_edge = -((mi_col as i32) * MI_SIZE * 8);
    let mb_to_right_edge = ((mi_cols as i32) - (bw as i32) - (mi_col as i32)) * MI_SIZE * 8;
    mvec.clamp(mb_to_left_edge - border, mb_to_right_edge + border)
}

/// `add_mv_ref_list( refList )` (spec §6.5.6). Handles appending to
/// `RefListMv`/`RefMvCount`, including deduplication (a value equal to
/// `RefListMv[0]` is not added) and capping at a maximum of 2 entries.
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

/// `scale_mv( refList, refFrame )` (spec §6.5.9). Flips the sign of the
/// motion vector if `ref_frame_sign_bias` differs between the candidate
/// frame and the target reference frame.
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

/// `use_mv_hp( deltaMv )` (spec §6.5.13).
pub fn use_mv_hp(delta_mv: Mv) -> bool {
    (delta_mv[0].abs() >> 3) < COMPANDED_MVREF_THRESH
        && (delta_mv[1].abs() >> 3) < COMPANDED_MVREF_THRESH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_mv_row_matches_spec_formula() {
        // An 8x8 block with MiRow=0, bh=1, MiRows=8 (64px frame).
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
        add_mv_ref_list(&mut list, &mut count, [1, 2]); // duplicate -> not added
        assert_eq!(count, 1);
        add_mv_ref_list(&mut list, &mut count, [3, 4]);
        assert_eq!(count, 2);
        assert_eq!(list, [[1, 2], [3, 4]]);
        add_mv_ref_list(&mut list, &mut count, [5, 6]); // ignored due to cap of 2
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
