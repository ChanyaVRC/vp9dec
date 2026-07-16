//! Mode info decoding: `intra_frame_mode_info`/`inter_frame_mode_info` (spec §6.4.6, §6.4.11)
//! and everything they call directly -- segment id (spec §6.4.7/§6.4.9/§6.4.12/§6.4.14),
//! `skip`/`tx_size` (spec §6.4.8/§6.4.10), `is_inter`/`ref_frames` (spec §6.4.13/§6.4.17), and
//! motion vector syntax (spec §6.4.18-6.4.20). Reference-frame neighbor-context derivation
//! (`comp_mode_ctx` and friends) lives in `tile::ref_ctx`; motion vector *prediction*
//! (`find_mv_refs` and friends) lives in `tile::mv_pred`.

use super::mv_pred::{use_mv_hp, Mv, ZERO_MV};
use super::{MiInfo, TileDecoder, TileError};
use crate::bool_coder::BoolDecoder;
use crate::common::INTRA_FRAME;
use crate::header::{SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use crate::prob_tables::{
    ALTREF_FRAME, BLOCK_8X8, COMPOUND_REFERENCE, DC_PRED, GOLDEN_FRAME, INTERP_FILTER_TREE,
    INTER_MODE_TREE, INTRA_MODE_TREE, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, LAST_FRAME,
    MAX_TXSIZE_LOOKUP, MV_CLASS_TREE, MV_FR_TREE, MV_JOINT_HNZVNZ, MV_JOINT_HNZVZ, MV_JOINT_HZVNZ,
    MV_JOINT_TREE, NEARESTMV, NEARMV, NEWMV, NUM_4X4_BLOCKS_HIGH_LOOKUP,
    NUM_4X4_BLOCKS_WIDE_LOOKUP, NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP,
    REFERENCE_MODE_SELECT, REF_NONE, SEGMENT_TREE, SINGLE_REFERENCE, SIZE_GROUP_LOOKUP, SWITCHABLE,
    TX_16X16, TX_32X32, TX_MODE_SELECT, TX_MODE_TO_BIGGEST_TX_SIZE, TX_SIZE_16_TREE,
    TX_SIZE_32_TREE, TX_SIZE_8_TREE, ZEROMV,
};

/// Neighbor information derived by `inter_frame_mode_info( )` (spec §6.4.11)
/// (`LeftRefFrame`/`AboveRefFrame`/`LeftIntra`/`AboveIntra`/`LeftSingle`/`AboveSingle`).
/// `pub(super)` (rather than private): read by `tile::ref_ctx`'s context-derivation methods.
pub(super) struct NeighborRefInfo {
    pub(super) avail_u: bool,
    pub(super) avail_l: bool,
    pub(super) left_ref_frame: [u8; 2],
    pub(super) above_ref_frame: [u8; 2],
    pub(super) left_intra: bool,
    pub(super) above_intra: bool,
    pub(super) left_single: bool,
    pub(super) above_single: bool,
}

impl TileDecoder {
    /// `seg_feature_active( feature )` (spec §6.4.9).
    fn seg_feature_active(&self, segment_id: u8, feature: usize) -> bool {
        self.segmentation.enabled && self.segmentation.feature_enabled[segment_id as usize][feature]
    }

    /// `intra_segment_id( )` (spec §6.4.7). Used for `FrameIsIntra` blocks
    /// (key frames / intra-only frames), which have no temporal prediction of
    /// `segment_id` (unlike `inter_segment_id`).
    fn intra_segment_id(&self, r: &mut BoolDecoder) -> u8 {
        if self.segmentation.enabled && self.segmentation.update_map {
            r.read_tree(&SEGMENT_TREE, |node| self.segmentation.tree_probs[node]) as u8
        } else {
            0
        }
    }

    /// `get_segment_id( )` (spec §6.4.14). The predicted segment id is the
    /// smallest value found in the on-screen region of `PrevSegmentIds`
    /// covered by the current block.
    fn get_segment_id(&self, row: u32, col: u32, mi_size: u8) -> u8 {
        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
        let xmis = (self.mi_cols - col).min(bw);
        let ymis = (self.mi_rows - row).min(bh);
        let mut seg = 7u8;
        for y in 0..ymis {
            for x in 0..xmis {
                let idx = ((row + y) * self.mi_cols + (col + x)) as usize;
                seg = seg.min(self.prev_segment_ids[idx]);
            }
        }
        seg
    }

    /// `inter_segment_id( )` (spec §6.4.12). Used for blocks in an inter frame
    /// (`!FrameIsIntra`), whether or not the individual block itself is inter-coded.
    fn inter_segment_id(&mut self, r: &mut BoolDecoder, row: u32, col: u32, mi_size: u8) -> u8 {
        if !self.segmentation.enabled {
            return 0;
        }
        let predicted_segment_id = self.get_segment_id(row, col, mi_size);
        if !self.segmentation.update_map {
            return predicted_segment_id;
        }
        if !self.segmentation.temporal_update {
            return r.read_tree(&SEGMENT_TREE, |node| self.segmentation.tree_probs[node]) as u8;
        }

        let ctx = (self.left_seg_pred_context[(row % 8) as usize]
            + self.above_seg_pred_context[col as usize]) as usize;
        let seg_id_predicted = r.read_bool(self.segmentation.pred_prob[ctx]);
        let segment_id = if seg_id_predicted {
            predicted_segment_id
        } else {
            r.read_tree(&SEGMENT_TREE, |node| self.segmentation.tree_probs[node]) as u8
        };

        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
        for i in 0..bw {
            self.above_seg_pred_context[(col + i) as usize] = seg_id_predicted as u8;
        }
        for i in 0..bh {
            self.left_seg_pred_context[((row + i) % 8) as usize] = seg_id_predicted as u8;
        }
        segment_id
    }

    /// `read_skip( )` (spec §6.4.8).
    fn read_skip(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        avail_u: bool,
        avail_l: bool,
        segment_id: u8,
    ) -> bool {
        if self.seg_feature_active(segment_id, SEG_LVL_SKIP) {
            return true;
        }
        let mut ctx = 0usize;
        if avail_u && self.mi_grid.get(row - 1, col).skip {
            ctx += 1;
        }
        if avail_l && self.mi_grid.get(row, col - 1).skip {
            ctx += 1;
        }
        let skip = r.read_bool(self.probs.skip_prob[ctx]);
        if !self.frame_is_intra {
            self.counts.skip[ctx][skip as usize] += 1;
        }
        skip
    }

    /// `read_tx_size( allowSelect )` (spec §6.4.10).
    fn read_tx_size(
        &mut self,
        r: &mut BoolDecoder,
        mi_size: u8,
        allow_select: bool,
        pos: (u32, u32),
        avail: (bool, bool),
    ) -> u8 {
        let (row, col) = pos;
        let (avail_u, avail_l) = avail;
        let max_tx_size = MAX_TXSIZE_LOOKUP[mi_size as usize];
        if allow_select && self.tx_mode == TX_MODE_SELECT && mi_size >= BLOCK_8X8 {
            let mut above = max_tx_size;
            let mut left = max_tx_size;
            if avail_u {
                let n = self.mi_grid.get(row - 1, col);
                if !n.skip {
                    above = n.tx_size;
                }
            }
            if avail_l {
                let n = self.mi_grid.get(row, col - 1);
                if !n.skip {
                    left = n.tx_size;
                }
            }
            if !avail_l {
                left = above;
            }
            if !avail_u {
                above = left;
            }
            let ctx = ((above as u32 + left as u32) > max_tx_size as u32) as usize;
            let probs = self.probs.tx_probs[max_tx_size as usize][ctx];
            let tx_size = match max_tx_size {
                TX_32X32 => r.read_tree(&TX_SIZE_32_TREE, |node| probs[node]) as u8,
                TX_16X16 => r.read_tree(&TX_SIZE_16_TREE, |node| probs[node]) as u8,
                _ => r.read_tree(&TX_SIZE_8_TREE, |node| probs[node]) as u8,
            };
            if !self.frame_is_intra {
                self.counts.tx_size[max_tx_size as usize][ctx][tx_size as usize] += 1;
            }
            tx_size
        } else {
            max_tx_size.min(TX_MODE_TO_BIGGEST_TX_SIZE[self.tx_mode as usize])
        }
    }

    /// `intra_frame_mode_info( )` (spec §6.4.6).
    pub(super) fn intra_frame_mode_info(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        mi_size: u8,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<MiInfo, TileError> {
        let segment_id = self.intra_segment_id(r);
        let skip = self.read_skip(r, row, col, avail_u, avail_l, segment_id);
        let tx_size = self.read_tx_size(r, mi_size, true, (row, col), (avail_u, avail_l));

        let mut sub_modes = [DC_PRED; 4];
        let y_mode;
        if mi_size >= BLOCK_8X8 {
            let above_mode = if avail_u {
                self.mi_grid.get(row - 1, col).sub_modes[2]
            } else {
                DC_PRED
            };
            let left_mode = if avail_l {
                self.mi_grid.get(row, col - 1).sub_modes[1]
            } else {
                DC_PRED
            };
            let mode = r.read_tree(&INTRA_MODE_TREE, |node| {
                KF_Y_MODE_PROBS[above_mode as usize][left_mode as usize][node]
            }) as u8;
            y_mode = mode;
            sub_modes = [mode; 4];
        } else {
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
            let mut last_mode = DC_PRED;
            let mut idy = 0u32;
            while idy < 2 {
                let mut idx = 0u32;
                while idx < 2 {
                    let above_mode = if idy > 0 {
                        sub_modes[idx as usize]
                    } else if avail_u {
                        self.mi_grid.get(row - 1, col).sub_modes[(2 + idx) as usize]
                    } else {
                        DC_PRED
                    };
                    let left_mode = if idx > 0 {
                        sub_modes[(idy * 2) as usize]
                    } else if avail_l {
                        self.mi_grid.get(row, col - 1).sub_modes[(1 + idy * 2) as usize]
                    } else {
                        DC_PRED
                    };
                    let mode = r.read_tree(&INTRA_MODE_TREE, |node| {
                        KF_Y_MODE_PROBS[above_mode as usize][left_mode as usize][node]
                    }) as u8;
                    for y2 in 0..num4x4h {
                        for x2 in 0..num4x4w {
                            sub_modes[((idy + y2) * 2 + idx + x2) as usize] = mode;
                        }
                    }
                    last_mode = mode;
                    idx += num4x4w;
                }
                idy += num4x4h;
            }
            y_mode = last_mode;
        }

        let uv_mode = r.read_tree(&INTRA_MODE_TREE, |node| {
            KF_UV_MODE_PROBS[y_mode as usize][node]
        }) as u8;

        Ok(MiInfo {
            skip,
            tx_size,
            mi_size,
            y_mode,
            uv_mode,
            sub_modes,
            segment_id,
            ref_frame: [INTRA_FRAME, REF_NONE],
            mv: [[0, 0]; 2],
            sub_mvs: [[[0, 0]; 4]; 2],
            interp_filter: 0,
        })
    }

    // =========================================================================
    // Inter frame (`FrameIsIntra == 0`) mode info decoding (spec §6.4.11-6.4.20).
    // =========================================================================

    /// `inter_frame_mode_info( )` (spec §6.4.11).
    pub(super) fn inter_frame_mode_info(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        mi_size: u8,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<MiInfo, TileError> {
        let left_ref_frame = if avail_l {
            self.mi_grid.get(row, col - 1).ref_frame
        } else {
            [INTRA_FRAME, REF_NONE]
        };
        let above_ref_frame = if avail_u {
            self.mi_grid.get(row - 1, col).ref_frame
        } else {
            [INTRA_FRAME, REF_NONE]
        };
        let neighbors = NeighborRefInfo {
            avail_u,
            avail_l,
            left_ref_frame,
            above_ref_frame,
            left_intra: left_ref_frame[0] == INTRA_FRAME,
            above_intra: above_ref_frame[0] == INTRA_FRAME,
            left_single: left_ref_frame[1] == REF_NONE,
            above_single: above_ref_frame[1] == REF_NONE,
        };

        let segment_id = self.inter_segment_id(r, row, col, mi_size);
        let skip = self.read_skip(r, row, col, avail_u, avail_l, segment_id);
        let is_inter = self.read_is_inter(r, &neighbors, segment_id);
        let tx_size = self.read_tx_size(
            r,
            mi_size,
            !skip || !is_inter,
            (row, col),
            (avail_u, avail_l),
        );

        if is_inter {
            self.inter_block_mode_info(r, row, col, mi_size, tx_size, skip, segment_id, &neighbors)
        } else {
            self.intra_block_mode_info(r, mi_size, tx_size, skip, segment_id)
        }
    }

    /// `read_is_inter( )` (spec §6.4.13).
    fn read_is_inter(&mut self, r: &mut BoolDecoder, n: &NeighborRefInfo, segment_id: u8) -> bool {
        if self.seg_feature_active(segment_id, SEG_LVL_REF_FRAME) {
            return self.segmentation.feature_data[segment_id as usize][SEG_LVL_REF_FRAME]
                != INTRA_FRAME as i32;
        }
        let ctx = if n.avail_u && n.avail_l {
            if n.left_intra && n.above_intra {
                3
            } else {
                (n.left_intra || n.above_intra) as usize
            }
        } else if n.avail_u || n.avail_l {
            2 * (if n.avail_u {
                n.above_intra
            } else {
                n.left_intra
            } as usize)
        } else {
            0
        };
        let is_inter = r.read_bool(self.probs.is_inter_prob[ctx]);
        self.counts.is_inter[ctx][is_inter as usize] += 1;
        is_inter
    }

    /// `intra_block_mode_info( )` (spec §6.4.15). For intra blocks within an inter frame.
    fn intra_block_mode_info(
        &mut self,
        r: &mut BoolDecoder,
        mi_size: u8,
        tx_size: u8,
        skip: bool,
        segment_id: u8,
    ) -> Result<MiInfo, TileError> {
        let mut sub_modes = [DC_PRED; 4];
        let y_mode;
        if mi_size >= BLOCK_8X8 {
            let ctx = SIZE_GROUP_LOOKUP[mi_size as usize] as usize;
            let mode =
                r.read_tree(&INTRA_MODE_TREE, |node| self.probs.y_mode_probs[ctx][node]) as u8;
            self.counts.intra_mode[ctx][mode as usize] += 1;
            y_mode = mode;
            sub_modes = [mode; 4];
        } else {
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
            let mut last_mode = DC_PRED;
            let mut idy = 0u32;
            while idy < 2 {
                let mut idx = 0u32;
                while idx < 2 {
                    // sub_intra_mode: ctx is always 0 (spec §9.3.2).
                    let mode = r
                        .read_tree(&INTRA_MODE_TREE, |node| self.probs.y_mode_probs[0][node])
                        as u8;
                    self.counts.intra_mode[0][mode as usize] += 1;
                    for y2 in 0..num4x4h {
                        for x2 in 0..num4x4w {
                            sub_modes[((idy + y2) * 2 + idx + x2) as usize] = mode;
                        }
                    }
                    last_mode = mode;
                    idx += num4x4w;
                }
                idy += num4x4h;
            }
            y_mode = last_mode;
        }

        let uv_mode = r.read_tree(&INTRA_MODE_TREE, |node| {
            self.probs.uv_mode_probs[y_mode as usize][node]
        }) as u8;
        self.counts.uv_mode[y_mode as usize][uv_mode as usize] += 1;

        Ok(MiInfo {
            skip,
            tx_size,
            mi_size,
            y_mode,
            uv_mode,
            sub_modes,
            segment_id,
            ref_frame: [INTRA_FRAME, REF_NONE],
            mv: [[0, 0]; 2],
            sub_mvs: [[[0, 0]; 4]; 2],
            interp_filter: 0,
        })
    }

    /// `read_ref_frames( )` (spec §6.4.17).
    fn read_ref_frames(
        &mut self,
        r: &mut BoolDecoder,
        n: &NeighborRefInfo,
        segment_id: u8,
    ) -> [u8; 2] {
        if self.seg_feature_active(segment_id, SEG_LVL_REF_FRAME) {
            return [
                self.segmentation.feature_data[segment_id as usize][SEG_LVL_REF_FRAME] as u8,
                REF_NONE,
            ];
        }
        let comp_mode = if self.reference_mode == REFERENCE_MODE_SELECT {
            let ctx = self.comp_mode_ctx(n);
            let bit = r.read_bool(self.probs.comp_mode_prob[ctx]);
            self.counts.comp_mode[ctx][bit as usize] += 1;
            (bit as u8) + SINGLE_REFERENCE
        } else {
            self.reference_mode
        };

        if comp_mode == COMPOUND_REFERENCE {
            let idx = self.ref_frame_sign_bias[self.comp_fixed_ref as usize] as usize;
            let ctx = self.comp_ref_ctx(n);
            let comp_ref = r.read_bool(self.probs.comp_ref_prob[ctx]) as usize;
            self.counts.comp_ref[ctx][comp_ref] += 1;
            let mut ref_frame = [0u8; 2];
            ref_frame[idx] = self.comp_fixed_ref;
            ref_frame[1 - idx] = self.comp_var_ref[comp_ref];
            ref_frame
        } else {
            let ctx1 = self.single_ref_p1_ctx(n);
            let single_ref_p1 = r.read_bool(self.probs.single_ref_prob[ctx1][0]);
            self.counts.single_ref[ctx1][0][single_ref_p1 as usize] += 1;
            if single_ref_p1 {
                let ctx2 = self.single_ref_p2_ctx(n);
                let single_ref_p2 = r.read_bool(self.probs.single_ref_prob[ctx2][1]);
                self.counts.single_ref[ctx2][1][single_ref_p2 as usize] += 1;
                [
                    if single_ref_p2 {
                        ALTREF_FRAME
                    } else {
                        GOLDEN_FRAME
                    },
                    REF_NONE,
                ]
            } else {
                [LAST_FRAME, REF_NONE]
            }
        }
    }

    /// Context derivation for `interp_filter` (spec §9.3.2). `3` is a sentinel value meaning
    /// "at least one of the two neighboring blocks is intra, or the filters disagree".
    fn interp_filter_ctx(&self, row: u32, col: u32, n: &NeighborRefInfo) -> usize {
        let left_interp = if n.avail_l && n.left_ref_frame[0] > INTRA_FRAME {
            self.mi_grid.get(row, col - 1).interp_filter
        } else {
            3
        };
        let above_interp = if n.avail_u && n.above_ref_frame[0] > INTRA_FRAME {
            self.mi_grid.get(row - 1, col).interp_filter
        } else {
            3
        };
        if left_interp == above_interp {
            left_interp as usize
        } else if left_interp == 3 && above_interp != 3 {
            above_interp as usize
        } else if left_interp != 3 && above_interp == 3 {
            left_interp as usize
        } else {
            3
        }
    }

    /// `inter_block_mode_info( )` (spec §6.4.16).
    #[allow(clippy::too_many_arguments)]
    fn inter_block_mode_info(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        mi_size: u8,
        tx_size: u8,
        skip: bool,
        segment_id: u8,
        n: &NeighborRefInfo,
    ) -> Result<MiInfo, TileError> {
        let ref_frame = self.read_ref_frames(r, n, segment_id);

        let mut nearest_mv: [Mv; 2] = [ZERO_MV; 2];
        let mut near_mv: [Mv; 2] = [ZERO_MV; 2];
        let mut best_mv: [Mv; 2] = [ZERO_MV; 2];
        let mut mode_context = [0u8; 4];

        for j in 0..2 {
            if ref_frame[j] > INTRA_FRAME {
                let (ref_list_mv, ctx) = self.find_mv_refs(row, col, mi_size, ref_frame[j], -1);
                mode_context[ref_frame[j] as usize] = ctx;
                let (nearest, near, best) = self.find_best_ref_mvs(row, col, mi_size, ref_list_mv);
                nearest_mv[j] = nearest;
                near_mv[j] = near;
                best_mv[j] = best;
            }
        }

        let is_compound = ref_frame[1] > INTRA_FRAME;
        let n_refs = 1 + is_compound as usize;

        // §6.4.16: when seg_feature_active(SEG_LVL_SKIP), y_mode is forced to ZEROMV without
        // reading inter_mode. Bitstream conformance guarantees MiSize >= BLOCK_8X8 whenever
        // seg_feature_active(SEG_LVL_SKIP) is set here, so the MiSize < BLOCK_8X8 sub8x8 loop
        // below never runs in that case.
        let mut y_mode = ZEROMV;
        if self.seg_feature_active(segment_id, SEG_LVL_SKIP) {
            // y_mode stays ZEROMV.
        } else if mi_size >= BLOCK_8X8 {
            let ctx = mode_context[ref_frame[0] as usize] as usize;
            let inter_mode = r.read_tree(&INTER_MODE_TREE, |node| {
                self.probs.inter_mode_probs[ctx][node]
            }) as u8;
            self.counts.inter_mode[ctx][inter_mode as usize] += 1;
            y_mode = NEARESTMV + inter_mode;
        }

        let interp_filter = if self.interpolation_filter == SWITCHABLE {
            let ctx = self.interp_filter_ctx(row, col, n);
            let f = r.read_tree(&INTERP_FILTER_TREE, |node| {
                self.probs.interp_filter_probs[ctx][node]
            }) as u8;
            self.counts.interp_filter[ctx][f as usize] += 1;
            f
        } else {
            self.interpolation_filter
        };

        let mut block_mvs = [[[0i32; 2]; 4]; 2];

        if mi_size < BLOCK_8X8 {
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
            let mut idy = 0u32;
            while idy < 2 {
                let mut idx = 0u32;
                while idx < 2 {
                    let ctx = mode_context[ref_frame[0] as usize] as usize;
                    let inter_mode = r.read_tree(&INTER_MODE_TREE, |node| {
                        self.probs.inter_mode_probs[ctx][node]
                    }) as u8;
                    self.counts.inter_mode[ctx][inter_mode as usize] += 1;
                    y_mode = NEARESTMV + inter_mode;
                    let block = (idy * 2 + idx) as i32;
                    if y_mode == NEARESTMV || y_mode == NEARMV {
                        for j in 0..n_refs {
                            let (nm, nr) = self.append_sub8x8_mvs(
                                row,
                                col,
                                mi_size,
                                block,
                                ref_frame[j],
                                j,
                                &block_mvs,
                            );
                            nearest_mv[j] = nm;
                            near_mv[j] = nr;
                        }
                    }
                    let mv = self.assign_mv(r, y_mode, n_refs, nearest_mv, near_mv, best_mv);
                    for y2 in 0..num4x4h {
                        for x2 in 0..num4x4w {
                            let b = ((idy + y2) * 2 + idx + x2) as usize;
                            for (rl, block_mv) in mv.iter().enumerate().take(n_refs) {
                                block_mvs[rl][b] = *block_mv;
                            }
                        }
                    }
                    idx += num4x4w;
                }
                idy += num4x4h;
            }
        } else {
            let mv = self.assign_mv(r, y_mode, n_refs, nearest_mv, near_mv, best_mv);
            for (rl, block_mv) in mv.iter().enumerate().take(n_refs) {
                for b in block_mvs[rl].iter_mut() {
                    *b = *block_mv;
                }
            }
        }

        Ok(MiInfo {
            skip,
            tx_size,
            mi_size,
            y_mode,
            uv_mode: 0,
            sub_modes: [DC_PRED; 4],
            segment_id,
            ref_frame,
            mv: [block_mvs[0][3], block_mvs[1][3]],
            sub_mvs: block_mvs,
            interp_filter,
        })
    }

    /// `assign_mv( isCompound )` (spec §6.4.18).
    fn assign_mv(
        &mut self,
        r: &mut BoolDecoder,
        y_mode: u8,
        n_refs: usize,
        nearest_mv: [Mv; 2],
        near_mv: [Mv; 2],
        best_mv: [Mv; 2],
    ) -> [Mv; 2] {
        let mut mv = [ZERO_MV; 2];
        for (i, slot) in mv.iter_mut().enumerate().take(n_refs) {
            *slot = match y_mode {
                NEWMV => self.read_mv(r, best_mv[i]),
                NEARESTMV => nearest_mv[i],
                NEARMV => near_mv[i],
                _ => ZERO_MV, // ZEROMV
            };
        }
        mv
    }

    /// `read_mv( ref )` (spec §6.4.19).
    fn read_mv(&mut self, r: &mut BoolDecoder, best_mv: Mv) -> Mv {
        let use_hp = self.allow_high_precision_mv && use_mv_hp(best_mv);
        let mv_joint = r.read_tree(&MV_JOINT_TREE, |node| self.probs.mv_joint_probs[node]) as u8;
        self.counts.mv_joint[mv_joint as usize] += 1;
        let mut diff = ZERO_MV;
        if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
            diff[0] = self.read_mv_component(r, 0, use_hp);
        }
        if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
            diff[1] = self.read_mv_component(r, 1, use_hp);
        }
        [best_mv[0] + diff[0], best_mv[1] + diff[1]]
    }

    /// `read_mv_component( comp )` (spec §6.4.20).
    fn read_mv_component(&mut self, r: &mut BoolDecoder, comp: usize, use_hp: bool) -> i32 {
        let sign = r.read_bool(self.probs.mv_sign_prob[comp]);
        self.counts.mv_sign[comp][sign as usize] += 1;
        let mv_class =
            r.read_tree(&MV_CLASS_TREE, |node| self.probs.mv_class_probs[comp][node]) as usize;
        self.counts.mv_class[comp][mv_class] += 1;
        let mag: u32 = if mv_class == 0 {
            let class0_bit = r.read_bool(self.probs.mv_class0_bit_prob[comp]) as u32;
            self.counts.mv_class0_bit[comp][class0_bit as usize] += 1;
            let class0_fr = r.read_tree(&MV_FR_TREE, |node| {
                self.probs.mv_class0_fr_probs[comp][class0_bit as usize][node]
            }) as u32;
            self.counts.mv_class0_fr[comp][class0_bit as usize][class0_fr as usize] += 1;
            let class0_hp = if use_hp {
                r.read_bool(self.probs.mv_class0_hp_prob[comp]) as u32
            } else {
                1
            };
            self.counts.mv_class0_hp[comp][class0_hp as usize] += 1;
            ((class0_bit << 3) | (class0_fr << 1) | class0_hp) + 1
        } else {
            let mut d: u32 = 0;
            for i in 0..mv_class {
                let bit = r.read_bool(self.probs.mv_bits_prob[comp][i]) as u32;
                self.counts.mv_bits[comp][i][bit as usize] += 1;
                d |= bit << i;
            }
            let mut mag = 2u32 << (mv_class + 2); // CLASS0_SIZE(2) << (mv_class+2)
            let fr = r.read_tree(&MV_FR_TREE, |node| self.probs.mv_fr_probs[comp][node]) as u32;
            self.counts.mv_fr[comp][fr as usize] += 1;
            let hp = if use_hp {
                r.read_bool(self.probs.mv_hp_prob[comp]) as u32
            } else {
                1
            };
            self.counts.mv_hp[comp][hp as usize] += 1;
            mag += ((d << 3) | (fr << 1) | hp) + 1;
            mag
        };
        if sign {
            -(mag as i32)
        } else {
            mag as i32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed_header::{CompressedHeader, CompressedHeaderProbs};
    use crate::counts::Counts;
    use crate::header::{
        ColorConfig, FrameType, LoopFilterParams, NewFrameHeader, QuantizationParams,
    };
    use crate::prob_tables::{BLOCK_32X32, BLOCK_8X8, ONLY_4X4};
    use crate::test_support::BoolEncoder;
    use std::sync::Arc;

    /// A disabled `SegmentationParams` (the M2 default / most existing tests).
    fn no_segmentation() -> crate::header::SegmentationParams {
        crate::header::SegmentationParams {
            enabled: false,
            update_map: false,
            tree_probs: [255; 7],
            pred_prob: [255; 3],
            temporal_update: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; 4]; 8],
            feature_data: [[0; 4]; 8],
        }
    }

    /// Builds a minimal `NewFrameHeader` for tests. An 8x8 (1 MI, 1 SB) key frame.
    fn minimal_header(width: u32, height: u32) -> NewFrameHeader {
        NewFrameHeader {
            profile: 0,
            frame_type: FrameType::KeyFrame,
            show_frame: true,
            error_resilient_mode: false,
            frame_is_intra: true,
            intra_only: false,
            reset_frame_context: 0,
            ref_frame_idx: [0, 0, 0],
            ref_frame_sign_bias: [false; 4],
            allow_high_precision_mv: false,
            interpolation_filter: crate::prob_tables::SWITCHABLE,
            color_config: Some(ColorConfig {
                bit_depth: 8,
                color_space: 0,
                color_range: false,
                subsampling_x: 1,
                subsampling_y: 1,
            }),
            width,
            height,
            render_width: width,
            render_height: height,
            refresh_frame_flags: 0xFF,
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            frame_context_idx_raw: 0,
            loop_filter: LoopFilterParams {
                level: 0,
                sharpness: 0,
                delta_enabled: false,
                ref_deltas: [1, 0, -1, -1],
                mode_deltas: [0, 0],
            },
            quantization: QuantizationParams {
                base_q_idx: 0,
                delta_q_y_dc: 0,
                delta_q_uv_dc: 0,
                delta_q_uv_ac: 0,
                lossless: true,
            },
            segmentation: no_segmentation(),
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            header_size_in_bytes: 0,
        }
    }

    /// Builds a minimal inter (non-intra-only) frame uncompressed header.
    fn minimal_inter_header(width: u32, height: u32) -> NewFrameHeader {
        let mut h = minimal_header(width, height);
        h.frame_type = FrameType::NonKeyFrame;
        h.frame_is_intra = false;
        h.ref_frame_idx = [0, 1, 2];
        h
    }

    fn default_compressed_header() -> CompressedHeader {
        CompressedHeader {
            tx_mode: ONLY_4X4,
            probs: Arc::new(CompressedHeaderProbs::default()),
            reference_mode: SINGLE_REFERENCE,
            comp_fixed_ref: 0,
            comp_var_ref: [0, 0],
        }
    }

    /// `intra_segment_id()` reads `segment_tree` when `update_map == 1`.
    #[test]
    fn intra_segment_id_reads_tree_when_update_map() {
        let mut header = minimal_header(8, 8);
        header.segmentation.enabled = true;
        header.segmentation.update_map = true;
        header.segmentation.tree_probs = [128; 7];
        let compressed = default_compressed_header();
        let decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);

        // segment_tree[14] = {2,4,6,8,10,12, 0,-1,-2,-3,-4,-5,-6,-7} (spec §9.3.1). Leaf
        // value 5 is reached via node 0 (bit=1 -> index 4), node 2 (bit=0 -> index 10),
        // node 5 (bit=1 -> index 11, leaf -(-5)=5): bit path [1,0,1].
        let mut enc = BoolEncoder::new();
        enc.write_bool(true, 128);
        enc.write_bool(false, 128);
        enc.write_bool(true, 128);
        let buf = enc.finish();
        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");

        assert_eq!(decoder.intra_segment_id(&mut r), 5);
    }

    /// `intra_segment_id()` returns 0 without reading any bits when segmentation is
    /// disabled or `update_map == 0`.
    #[test]
    fn intra_segment_id_is_zero_without_update_map() {
        let header = minimal_header(8, 8); // segmentation disabled.
        let compressed = default_compressed_header();
        let decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);

        // An empty tile: if intra_segment_id tried to read a bit, this would panic/error.
        let mut r = BoolDecoder::new(&[0x00]).expect("valid bitstream");
        assert_eq!(decoder.intra_segment_id(&mut r), 0);
    }

    /// `get_segment_id()` (spec §6.4.14): the predicted id is the minimum over the
    /// on-screen `PrevSegmentIds` region covered by the block, clipped at the frame edge.
    #[test]
    fn get_segment_id_takes_min_over_block_region() {
        // 32x32 -> MiCols=MiRows=4. prev_segment_ids laid out row-major 4x4.
        let header = minimal_header(32, 32);
        let compressed = default_compressed_header();
        #[rustfmt::skip]
        let prev_segment_ids = vec![
            3, 3, 3, 3,
            3, 1, 2, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
        ];
        let decoder = TileDecoder::new_with_prev(
            &header,
            header.color_config.unwrap(),
            &compressed,
            false,
            None,
            [None, None, None],
            Arc::new(prev_segment_ids),
        );
        // BLOCK_32X32 at (0,0) covers the whole 4x4 region -> min is 1.
        assert_eq!(decoder.get_segment_id(0, 0, BLOCK_32X32), 1);
        // BLOCK_8X8 at (1,1) covers only PrevSegmentIds[1][1] = 1.
        assert_eq!(decoder.get_segment_id(1, 1, BLOCK_8X8), 1);
        // BLOCK_8X8 at (0,0) covers only PrevSegmentIds[0][0] = 3.
        assert_eq!(decoder.get_segment_id(0, 0, BLOCK_8X8), 3);
    }

    /// `inter_segment_id()` (spec §6.4.12): when `seg_id_predicted == 1`, `segment_id` is
    /// taken from `get_segment_id()` (the previous frame's map) without reading `segment_id`.
    #[test]
    fn inter_segment_id_temporal_prediction_uses_prev_map() {
        let mut header = minimal_inter_header(8, 8);
        header.segmentation.enabled = true;
        header.segmentation.update_map = true;
        header.segmentation.temporal_update = true;
        header.segmentation.pred_prob = [64; 3];
        let compressed = default_compressed_header();
        let prev_segment_ids = vec![6u8]; // MiCols=MiRows=1 at 8x8.
        let mut decoder = TileDecoder::new_with_prev(
            &header,
            header.color_config.unwrap(),
            &compressed,
            false,
            None,
            [None, None, None],
            Arc::new(prev_segment_ids),
        );

        // seg_id_predicted = 1: only 1 bit is read (no segment_id tree read).
        let mut enc = BoolEncoder::new();
        enc.write_bool(true, 64);
        let buf = enc.finish();
        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");

        assert_eq!(decoder.inter_segment_id(&mut r, 0, 0, BLOCK_8X8), 6);
    }

    /// `read_skip()` (spec §6.4.8): when `seg_feature_active( SEG_LVL_SKIP )`, `skip` is
    /// forced to 1 without reading a bit or incrementing `counts.skip`.
    #[test]
    fn read_skip_seg_lvl_skip_forces_without_reading_bit() {
        let mut header = minimal_header(8, 8);
        header.segmentation.enabled = true;
        header.segmentation.feature_enabled[2][SEG_LVL_SKIP] = true;
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);

        // Empty tile data: if read_skip tried to read a bit, BoolDecoder::new would still
        // succeed on an all-zero buffer, but the returned value would come from the (absent)
        // stream rather than being forced; counts.skip is the reliable signal here.
        let mut r = BoolDecoder::new(&[0x00]).expect("valid bitstream");
        let skip = decoder.read_skip(&mut r, 0, 0, false, false, 2);
        assert!(skip);
        assert_eq!(decoder.counts.skip, Counts::new().skip);
    }

    /// `read_is_inter()` (spec §6.4.13): when `seg_feature_active( SEG_LVL_REF_FRAME )`,
    /// `is_inter` is derived from `FeatureData` without reading a bit or counting.
    #[test]
    fn read_is_inter_seg_lvl_ref_frame_forces_without_reading_bit() {
        let mut header = minimal_inter_header(8, 8);
        header.segmentation.enabled = true;
        header.segmentation.feature_enabled[1][SEG_LVL_REF_FRAME] = true;
        header.segmentation.feature_data[1][SEG_LVL_REF_FRAME] = LAST_FRAME as i32;
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);
        let n = NeighborRefInfo {
            avail_u: false,
            avail_l: false,
            left_ref_frame: [INTRA_FRAME, REF_NONE],
            above_ref_frame: [INTRA_FRAME, REF_NONE],
            left_intra: true,
            above_intra: true,
            left_single: true,
            above_single: true,
        };

        let mut r = BoolDecoder::new(&[0x00]).expect("valid bitstream");
        assert!(decoder.read_is_inter(&mut r, &n, 1));
        assert_eq!(decoder.counts.is_inter, Counts::new().is_inter);
    }

    /// `read_ref_frames()` (spec §6.4.17): when `seg_feature_active( SEG_LVL_REF_FRAME )`,
    /// `ref_frame` is `[FeatureData, NONE]` (no compound) without reading a bit or counting.
    #[test]
    fn read_ref_frames_seg_lvl_ref_frame_returns_feature_value() {
        let mut header = minimal_inter_header(8, 8);
        header.segmentation.enabled = true;
        header.segmentation.feature_enabled[4][SEG_LVL_REF_FRAME] = true;
        header.segmentation.feature_data[4][SEG_LVL_REF_FRAME] = GOLDEN_FRAME as i32;
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);
        let n = NeighborRefInfo {
            avail_u: false,
            avail_l: false,
            left_ref_frame: [INTRA_FRAME, REF_NONE],
            above_ref_frame: [INTRA_FRAME, REF_NONE],
            left_intra: true,
            above_intra: true,
            left_single: true,
            above_single: true,
        };

        let mut r = BoolDecoder::new(&[0x00]).expect("valid bitstream");
        assert_eq!(
            decoder.read_ref_frames(&mut r, &n, 4),
            [GOLDEN_FRAME, REF_NONE]
        );
        assert_eq!(decoder.counts.comp_mode, Counts::new().comp_mode);
        assert_eq!(decoder.counts.single_ref, Counts::new().single_ref);
    }

    // =========================================================================
    // Unit tests for MV decoding (spec §6.4.19-6.4.20).
    // Encode a known MV with `BoolEncoder` (test_support), decode it with
    // `TileDecoder::read_mv`/`read_mv_component` (private methods), and check round-trip equality.
    // =========================================================================

    #[test]
    fn read_mv_component_class0_roundtrip() {
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);
        let probs = CompressedHeaderProbs::default();

        // mv_sign=0(positive), mv_class=MV_CLASS_0, class0_bit=1, class0_fr=2, (since
        // use_hp=false, class0_hp is not read and is taken as 1).
        // mag = ((1<<3)|(2<<1)|1) + 1 = 13 + 1 = 14
        let mut enc = BoolEncoder::new();
        enc.write_bool(false, probs.mv_sign_prob[0]);
        enc.write_bool(false, probs.mv_class_probs[0][0]); // MV_CLASS_0 (tree leaf)
        enc.write_bool(true, probs.mv_class0_bit_prob[0]); // class0_bit = 1
                                                           // class0_fr = 2: bit sequence [1,1,0] of MV_FR_TREE
        enc.write_bool(true, probs.mv_class0_fr_probs[0][1][0]);
        enc.write_bool(true, probs.mv_class0_fr_probs[0][1][1]);
        enc.write_bool(false, probs.mv_class0_fr_probs[0][1][2]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        let mag = decoder.read_mv_component(&mut r, 0, false);
        assert_eq!(mag, 14);
    }

    #[test]
    fn read_mv_component_negative_class0_roundtrip() {
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);
        let probs = CompressedHeaderProbs::default();

        let mut enc = BoolEncoder::new();
        enc.write_bool(true, probs.mv_sign_prob[1]); // sign = negative
        enc.write_bool(false, probs.mv_class_probs[1][0]); // MV_CLASS_0
        enc.write_bool(false, probs.mv_class0_bit_prob[1]); // class0_bit = 0
                                                            // class0_fr = 0: bit sequence [0]
        enc.write_bool(false, probs.mv_class0_fr_probs[1][0][0]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        // mag = ((0<<3)|(0<<1)|1) + 1 = 2, and since the sign is negative, -2.
        let mag = decoder.read_mv_component(&mut r, 1, false);
        assert_eq!(mag, -2);
    }

    #[test]
    fn read_mv_component_higher_class_roundtrip() {
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);
        let probs = CompressedHeaderProbs::default();

        // mv_class = MV_CLASS_1 (value 1): in the tree [0,2,-1,4,...], bit0=1,bit1=0 -> leaf -1 (value 1).
        let mut enc = BoolEncoder::new();
        enc.write_bool(false, probs.mv_sign_prob[0]); // positive
        enc.write_bool(true, probs.mv_class_probs[0][0]);
        enc.write_bool(false, probs.mv_class_probs[0][1]);
        // mv_class=1 -> read 1 bit of d (mv_bit). Let d=1.
        enc.write_bool(true, probs.mv_bits_prob[0][0]);
        // mv_fr = 1: bit sequence [1,0]
        enc.write_bool(true, probs.mv_fr_probs[0][0]);
        enc.write_bool(false, probs.mv_fr_probs[0][1]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        // mag = CLASS0_SIZE << (1+2) = 2<<3 = 16; d=1 (bit0=1), fr=1, hp(forced 1)
        // mag += ((1<<3)|(1<<1)|1) + 1 = (8|2|1)+1 = 11+1 = 12 -> total 16+12 = 28
        let mag = decoder.read_mv_component(&mut r, 0, false);
        assert_eq!(mag, 28);
    }

    #[test]
    fn read_mv_full_roundtrip_with_best_mv_offset() {
        // read_mv(ref) = BestMv + diffMv. mv_joint = MV_JOINT_HNZVNZ (both components nonzero).
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);
        let probs = CompressedHeaderProbs::default();
        let best_mv: Mv = [10, -20];

        let mut enc = BoolEncoder::new();
        // mv_joint tree: bit sequence [1,1,1] for MV_JOINT_HNZVNZ(=3)
        enc.write_bool(true, probs.mv_joint_probs[0]);
        enc.write_bool(true, probs.mv_joint_probs[1]);
        enc.write_bool(true, probs.mv_joint_probs[2]);
        // comp 0 (row): mag=14 (same class0 pattern as the earlier test, positive sign)
        enc.write_bool(false, probs.mv_sign_prob[0]);
        enc.write_bool(false, probs.mv_class_probs[0][0]);
        enc.write_bool(true, probs.mv_class0_bit_prob[0]);
        enc.write_bool(true, probs.mv_class0_fr_probs[0][1][0]);
        enc.write_bool(true, probs.mv_class0_fr_probs[0][1][1]);
        enc.write_bool(false, probs.mv_class0_fr_probs[0][1][2]);
        // comp 1 (col): mag=2, negative sign -> -2
        enc.write_bool(true, probs.mv_sign_prob[1]);
        enc.write_bool(false, probs.mv_class_probs[1][0]);
        enc.write_bool(false, probs.mv_class0_bit_prob[1]);
        enc.write_bool(false, probs.mv_class0_fr_probs[1][0][0]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        // allow_high_precision_mv=false, so use_hp is always false (regardless of use_mv_hp's result).
        let mv = decoder.read_mv(&mut r, best_mv);
        assert_eq!(mv, [10 + 14, -20 + (-2)]);
    }
}
