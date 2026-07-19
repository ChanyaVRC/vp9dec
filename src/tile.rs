//! Decoding of tiles, superblocks, mode info, and coefficients (tokens)
//! (spec §6.4 "Decode tiles syntax").
//!
//! M3 adds support for inter frames (`FrameIsIntra == 0`); `mode_info()` dispatches to
//! `intra_frame_mode_info()`/`inter_frame_mode_info()` depending on `frame_is_intra`.
//! `inter_frame_mode_info()` (spec §6.4.11-6.4.20) is implemented including motion vector
//! prediction (spec §6.5, `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs`), and
//! motion compensation / sub-pixel interpolation is performed via
//! [`predict_inter`](crate::predict::predict_inter) (spec §8.5.2).
//!
//! [`TileDecoder::residual`] implements `residual()` from spec §6.4.21, and for each plane
//! performs intra prediction ([`crate::predict::predict_intra`]) -> token decoding
//! ([`TileDecoder::tokens_and_reconstruct`], spec §6.4.24-6.4.26) -> inverse quantization,
//! inverse transform, and reconstruction (spec §8.6.2), writing the result into the frame
//! buffers in [`TileDecoder::planes`].
//! The loop filter (spec §8.8) is applied separately, via [`TileDecoder::apply_loop_filter`],
//! after all tiles in the frame have been decoded.
//!
//! W5 split this module's implementation across a `tile/` directory of submodules (this file
//! remains the hub: the `TileDecoder` struct/fields and the tile/superblock/partition/block
//! traversal that all submodules are ultimately driven by). The submodules are private (not
//! part of the crate's already-`#[doc(hidden)]` public surface), so referenced here by name
//! only, not as intra-doc links:
//! - `mode_info`: `intra_frame_mode_info`/`inter_frame_mode_info` and everything they call
//!   directly (segment id, skip, tx_size, is_inter, ref_frames, motion vector syntax).
//! - `ref_ctx`: pure neighbor-context derivation for reference-frame syntax elements.
//! - `mv_pred`: motion vector *prediction* (`find_mv_refs` and friends); also absorbs the
//!   former standalone `src/mv.rs`.
//! - `residual`: `residual()` and everything it calls directly (intra/inter prediction
//!   dispatch, token reading, inverse quantization/transform/reconstruction).

mod mode_info;
mod mv_pred;
mod ref_ctx;
mod residual;

pub use mv_pred::Mv;

use std::sync::Arc;

use crate::bool_coder::{BoolCoderError, BoolDecoder};
use crate::common::INTRA_FRAME;
use crate::compressed_header::{CompressedHeader, CompressedHeaderProbs};
use crate::counts::Counts;
use crate::dpb::RefFrameData;
use crate::framebuffer::Plane;
use crate::header::{self, ColorConfig, NewFrameHeader, SegmentationParams, MAX_SEGMENTS};
use crate::prob_tables::{
    BLOCK_64X64, BLOCK_8X8, BLOCK_INVALID, B_HEIGHT_LOG2_LOOKUP, B_WIDTH_LOG2_LOOKUP, DC_PRED,
    KF_PARTITION_PROBS, MI_WIDTH_LOG2_LOOKUP, NUM_8X8_BLOCKS_HIGH_LOOKUP,
    NUM_8X8_BLOCKS_WIDE_LOOKUP, PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_TREE,
    PARTITION_VERT, REF_NONE, SUBSIZE_LOOKUP, TX_4X4,
};

/// Errors that can occur while decoding tiles/partitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileError {
    /// Failed to initialize the bool decoder for the tile data.
    BoolCoder(BoolCoderError),
    /// Tile partitioning is invalid, e.g. the tile size field exceeds the data length.
    InvalidTileSize,
    /// `subsize_lookup` returned `BLOCK_INVALID` (bitstream inconsistency).
    InvalidPartition,
}

/// Information held by a single 8x8 mode info unit (spec §2.37 "Mode info").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiInfo {
    pub skip: bool,
    pub tx_size: u8,
    pub mi_size: u8,
    /// `y_mode` (same value as all of `sub_modes` when `MiSize >= BLOCK_8X8`). For intra
    /// blocks this is `DC_PRED`..`TM_PRED` (0..9); for inter blocks it is `NEARESTMV`..`NEWMV` (10..13).
    pub y_mode: u8,
    /// Only meaningful for intra blocks (unused, 0, for inter blocks).
    pub uv_mode: u8,
    /// Intra modes for the 4 4x4 sub-blocks within an 8x8 unit (can take multiple distinct
    /// values only when `MiSize < BLOCK_8X8`). Unused for inter blocks (`[DC_PRED; 4]`).
    pub sub_modes: [u8; 4],
    pub segment_id: u8,
    /// `ref_frame[ 0..2 ]` (spec §7.4.12). For intra blocks this is `[INTRA_FRAME, REF_NONE]`.
    /// `ref_frame[0] != INTRA_FRAME` is itself the condition for "this is an inter block"
    /// (`is_inter`).
    pub ref_frame: [u8; 2],
    /// `Mvs[ refList ]` (representative MV, same as `BlockMvs[ refList ][ 3 ]`, spec §6.4.4).
    /// Units are 1/8 pel. `[[0, 0]; 2]` for intra blocks.
    pub mv: [[i32; 2]; 2],
    /// `SubMvs[ refList ][ block 0..4 ]` (spec §6.4.4). Used by `get_sub_block_mv` in
    /// `find_mv_refs` when referencing the sub MVs of neighboring blocks.
    pub sub_mvs: [[[i32; 2]; 4]; 2],
    /// `interp_filter` (spec §6.4.16). Unused for intra blocks (0).
    pub interp_filter: u8,
}

impl Default for MiInfo {
    fn default() -> Self {
        Self {
            skip: false,
            tx_size: TX_4X4,
            mi_size: crate::prob_tables::BLOCK_4X4,
            y_mode: DC_PRED,
            uv_mode: DC_PRED,
            sub_modes: [DC_PRED; 4],
            segment_id: 0,
            ref_frame: [INTRA_FRAME, REF_NONE],
            mv: [[0, 0]; 2],
            sub_mvs: [[[0, 0]; 4]; 2],
            interp_filter: 0,
        }
    }
}

/// A grid holding mode info for the whole frame, in 8x8 units.
///
/// This bundles the spec's `Skips`/`TxSizes`/`MiSizes`/`YModes`/`SegmentIds`/`SubModes`
/// arrays into a single struct. Its size is `Sb64Cols*8 x Sb64Rows*8`
/// (allocated to also cover the portion of edge superblocks that extends past the frame).
///
/// Not `Clone`: the previous frame's grid is shared into the next frame's `TileDecoder` via
/// `Arc<MiGrid>` (see [`crate::Decoder::prev_mi_grid`]) rather than deep-cloned.
#[derive(Debug)]
pub struct MiGrid {
    cols: usize,
    rows: usize,
    data: Vec<MiInfo>,
}

impl MiGrid {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            data: vec![MiInfo::default(); cols * rows],
        }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn get(&self, row: u32, col: u32) -> &MiInfo {
        &self.data[row as usize * self.cols + col as usize]
    }

    fn get_mut(&mut self, row: u32, col: u32) -> &mut MiInfo {
        &mut self.data[row as usize * self.cols + col as usize]
    }
}

/// `get_tile_offset( tileNum, mis, tileSzLog2 )` (spec §6.4.1).
fn get_tile_offset(tile_num: u32, mis: u32, tile_sz_log2: u32) -> u32 {
    let sbs = (mis + 7) >> 3;
    let offset = ((tile_num * sbs) >> tile_sz_log2) << 3;
    offset.min(mis)
}

/// Decoder that walks tiles and superblocks to decode mode info.
///
/// Holds per-frame state (mode info grid, above/left partition context).
pub struct TileDecoder {
    tx_mode: u8,
    /// `Arc`-shared with the `CompressedHeader` that produced it (never mutated here).
    probs: Arc<CompressedHeaderProbs>,
    segmentation: SegmentationParams,
    mi_cols: u32,
    mi_rows: u32,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    mi_grid: MiGrid,
    /// `AbovePartitionContext`. Persists across the whole frame width (`Sb64Cols*8`) and is
    /// cleared only once per frame (at the start of `decode_tiles`) (spec §7.4.1).
    above_partition_context: Vec<u8>,
    /// `LeftPartitionContext`. Conceptually an array spanning the full frame height, but it
    /// is cleared per superblock row and only ever accessed at the relative position within
    /// that superblock row (absolute mi row number mod 8), so it is kept as an 8-element array.
    left_partition_context: [u8; 8],
    /// `AboveSegPredContext` (spec §6.4.12). Same persistence/sizing rationale as
    /// `above_partition_context`.
    above_seg_pred_context: Vec<u8>,
    /// `LeftSegPredContext` (spec §6.4.12). Same persistence/sizing rationale as
    /// `left_partition_context`.
    left_seg_pred_context: [u8; 8],
    /// `PrevSegmentIds[ MiRow ][ MiCol ]` (spec §6.4.14 `get_segment_id`), in
    /// row-major `MiRows x MiCols` layout (unpadded, unlike `mi_grid`). Supplied
    /// by the caller ([`Decoder`](crate::Decoder)), which is responsible for the
    /// size-change-clears-to-zero rule of spec §7.2.6. `Arc`-shared with the caller's
    /// copy (never mutated here) to avoid a clone-in every frame.
    prev_segment_ids: Arc<Vec<u8>>,
    // Range of the tile currently being decoded.
    mi_col_start: u32,
    mi_col_end: u32,
    mi_row_start: u32,
    mi_row_end: u32,

    // --- Additional state needed for coefficient decoding/reconstruction (spec §6.4.21-6.4.26, §8.5-8.6). ---
    bit_depth: u8,
    subsampling_x: u32,
    subsampling_y: u32,
    lossless: bool,
    /// Per-frame dequant step table (see [`residual::build_dequant_table`]); replaces
    /// re-deriving `get_qindex`/`get_dc_quant`/`get_ac_quant` on every transform block.
    dequant_table: [[[i64; 2]; 2]; MAX_SEGMENTS],
    /// `CurrFrame[ plane ]`. Index 0=Y, 1=U, 2=V.
    planes: [Plane; 3],
    /// `AboveNonzeroContext[ plane ]`. Persists across the whole frame width in 4x4 units
    /// (cleared only once per frame).
    above_nonzero_context: [Vec<u8>; 3],
    /// `LeftNonzeroContext[ plane ]`. Cleared per superblock row and only ever accessed at
    /// the relative position within that superblock row (absolute 4x4 row number mod 16),
    /// so it is kept as a 16-element array (64 pixels / 4 = 16).
    left_nonzero_context: [[u8; 16]; 3],

    // --- Additional state needed for inter frames (spec §6.4.11-6.4.20, §6.5). ---
    /// `FrameIsIntra`.
    frame_is_intra: bool,
    /// `ref_frame_sign_bias[ 0..4 ]` (from the uncompressed header).
    ref_frame_sign_bias: [bool; 4],
    allow_high_precision_mv: bool,
    /// `interpolation_filter` (when `SWITCHABLE`, `interp_filter` is read per block).
    interpolation_filter: u8,
    /// `reference_mode`/`CompFixedRef`/`CompVarRef` (from the compressed header, spec §6.3.12/6.3.18).
    reference_mode: u8,
    comp_fixed_ref: u8,
    comp_var_ref: [u8; 2],
    /// `UsePrevFrameMvs` (spec §7.2.6). When true, `Mvs`/`RefFrames` of the previous frame
    /// are referenced via `prev_mi_grid`.
    use_prev_frame_mvs: bool,
    /// `MiGrid` of the most recently decoded frame (equivalent to `PrevMvs`/`PrevRefFrames`).
    /// Not referenced when `use_prev_frame_mvs == false` (may be `None`). `Arc`-shared with
    /// the caller's copy (read-only here) to avoid a clone-in every frame.
    prev_mi_grid: Option<Arc<MiGrid>>,

    // --- Additional state needed for motion compensation (spec §8.5.2). ---
    /// `FrameWidth`/`FrameHeight` (actual size used for display/scaling calculations, distinct
    /// from the padded size such as `mi_cols*8`).
    frame_width: u32,
    frame_height: u32,
    /// Actual pixel data of the reference frames used to decode this frame (already resolved
    /// from the DPB by the caller using `header.ref_frame_idx`). Indexed by the `ref_frame`
    /// value `LAST_FRAME..=ALTREF_FRAME` minus `LAST_FRAME`. All elements are `None` when
    /// `FrameIsIntra == 1`. `Arc`-shared with the DPB slot(s) they were resolved from,
    /// rather than deep-cloned per reference.
    resolved_refs: [Option<Arc<RefFrameData>>; 3],

    // --- Counter collection for probability adaptation (spec §8.4, spec §9.3.4). ---
    counts: Counts,
}

impl TileDecoder {
    /// Builds a `TileDecoder` from the uncompressed and compressed headers.
    ///
    /// `use_prev_frame_mvs`/`prev_mi_grid` are always `false`/`None` (a simple constructor
    /// for key frames / M2 compatibility). `prev_segment_ids` is seeded to all-zero (the
    /// "first frame" state of spec §7.2.6), sized to this frame's `MiRows x MiCols`.
    pub fn new(
        header: &NewFrameHeader,
        color_config: ColorConfig,
        compressed: &CompressedHeader,
    ) -> Self {
        let image_size = header::compute_image_size(header.width, header.height);
        let zero_prev_segment_ids = Arc::new(vec![
            0u8;
            (image_size.mi_cols * image_size.mi_rows)
                as usize
        ]);
        Self::new_with_prev(
            header,
            color_config,
            compressed,
            false,
            None,
            [None, None, None],
            zero_prev_segment_ids,
        )
    }

    /// Inter-frame-capable version of [`TileDecoder::new`]. `color_config` is the resolved
    /// value for this frame (the caller -- `Decoder` -- resolves `NewFrameHeader::color_config`,
    /// which is `None` for a regular inter frame, before calling this; see `src/lib.rs`).
    /// `use_prev_frame_mvs`/`prev_mi_grid` correspond to `UsePrevFrameMvs` from spec §7.2.6 and
    /// the previous frame's `Mvs`/`RefFrames` that it references (the `usePrev` branch of
    /// `find_mv_refs`, spec §6.5.1). `resolved_refs` is the actual pixel data of the reference
    /// frames used for motion compensation (spec §8.5.2) (resolved by the caller from
    /// [`crate::dpb::Dpb`], indexed the same way as `ref_frame_idx`). `prev_segment_ids` is
    /// `PrevSegmentIds` (spec §6.4.14), row-major `MiRows x MiCols`; unlike `prev_mi_grid` it is
    /// NOT gated by `use_prev_frame_mvs` (its own persistence/reset rules are spec §7.2.6/§8.1
    /// step 3, tracked by the caller — see `Decoder::prev_segment_ids` in `src/lib.rs`).
    pub fn new_with_prev(
        header: &NewFrameHeader,
        color_config: ColorConfig,
        compressed: &CompressedHeader,
        use_prev_frame_mvs: bool,
        prev_mi_grid: Option<Arc<MiGrid>>,
        resolved_refs: [Option<Arc<RefFrameData>>; 3],
        prev_segment_ids: Arc<Vec<u8>>,
    ) -> Self {
        let image_size = header::compute_image_size(header.width, header.height);
        let grid_cols = (image_size.sb64_cols * 8) as usize;
        let grid_rows = (image_size.sb64_rows * 8) as usize;
        let subsampling_x = color_config.subsampling_x as u32;
        let subsampling_y = color_config.subsampling_y as u32;

        // Frame buffers are allocated rounded up to the superblock boundary
        // (Sb64Cols*64 / Sb64Rows*64) (see the docs in src/framebuffer.rs).
        let y_w = (image_size.sb64_cols * 64) as usize;
        let y_h = (image_size.sb64_rows * 64) as usize;
        let uv_w = y_w >> subsampling_x;
        let uv_h = y_h >> subsampling_y;
        let planes = [
            Plane::new(y_w, y_h),
            Plane::new(uv_w, uv_h),
            Plane::new(uv_w, uv_h),
        ];
        // AboveNonzeroContext persists across the whole frame width in 4x4 units (2x the 8x8 mi unit).
        let above_nz_len = grid_cols * 2;
        let above_nonzero_context = [
            vec![0u8; above_nz_len],
            vec![0u8; above_nz_len],
            vec![0u8; above_nz_len],
        ];

        Self {
            tx_mode: compressed.tx_mode,
            probs: compressed.probs.clone(),
            segmentation: header.segmentation,
            mi_cols: image_size.mi_cols,
            mi_rows: image_size.mi_rows,
            tile_cols_log2: header.tile_cols_log2,
            tile_rows_log2: header.tile_rows_log2,
            mi_grid: MiGrid::new(grid_cols, grid_rows),
            above_partition_context: vec![0u8; grid_cols],
            left_partition_context: [0u8; 8],
            above_seg_pred_context: vec![0u8; grid_cols],
            left_seg_pred_context: [0u8; 8],
            prev_segment_ids,
            mi_col_start: 0,
            mi_col_end: image_size.mi_cols,
            mi_row_start: 0,
            mi_row_end: image_size.mi_rows,
            bit_depth: color_config.bit_depth,
            subsampling_x,
            subsampling_y,
            lossless: header.quantization.lossless,
            dequant_table: residual::build_dequant_table(
                &header.segmentation,
                header.quantization.base_q_idx,
                color_config.bit_depth,
                header.quantization.delta_q_y_dc,
                header.quantization.delta_q_uv_dc,
                header.quantization.delta_q_uv_ac,
            ),
            planes,
            above_nonzero_context,
            left_nonzero_context: [[0u8; 16]; 3],
            frame_is_intra: header.frame_is_intra,
            ref_frame_sign_bias: header.ref_frame_sign_bias,
            allow_high_precision_mv: header.allow_high_precision_mv,
            interpolation_filter: header.interpolation_filter,
            reference_mode: compressed.reference_mode,
            comp_fixed_ref: compressed.comp_fixed_ref,
            comp_var_ref: compressed.comp_var_ref,
            use_prev_frame_mvs,
            prev_mi_grid,
            frame_width: header.width,
            frame_height: header.height,
            resolved_refs,
            counts: Counts::new(),
        }
    }

    /// The collected syntax element counters (used for probability adaptation, spec §8.4).
    pub fn counts(&self) -> &Counts {
        &self.counts
    }

    pub fn mi_grid(&self) -> &MiGrid {
        &self.mi_grid
    }

    /// Consumes `self` and hands back the finished `MiGrid` by value (no clone), for the
    /// caller to stash as next frame's `prev_mi_grid`. Must be the last thing done with this
    /// `TileDecoder` -- call any other accessor (`planes()`, `counts()`, etc.) first.
    pub fn into_mi_grid(self) -> MiGrid {
        self.mi_grid
    }

    /// Returns a reference to the decoded plane buffers (`CurrFrame`).
    /// Index 0=Y, 1=U, 2=V. The buffers have a size rounded up to the superblock boundary,
    /// so the caller must crop to the display size
    /// (done by [`crate::Decoder`]).
    pub fn planes(&self) -> &[Plane; 3] {
        &self.planes
    }

    /// Applies the loop filter (spec §8.8, [`crate::loop_filter`]) to the decoded `planes`.
    /// Call this right after `decode_tiles`, before reading the output via `planes()`.
    ///
    /// By directly borrowing `self.planes` (`&mut`) and `self.mi_grid` (`&`) as separate
    /// fields, both can be passed safely to `loop_filter_frame`, which needs both at once.
    pub fn apply_loop_filter(&mut self, lf: &header::LoopFilterParams) {
        crate::loop_filter::loop_filter_frame(
            &mut self.planes,
            &self.mi_grid,
            self.mi_cols,
            self.mi_rows,
            self.subsampling_x,
            self.subsampling_y,
            lf,
            &self.segmentation,
            self.bit_depth,
        );
    }

    /// `clear_above_context( )` (spec §7.4.1): also clears `AboveSegPredContext`.
    fn clear_above_context(&mut self) {
        self.above_partition_context.iter_mut().for_each(|v| *v = 0);
        self.above_seg_pred_context.iter_mut().for_each(|v| *v = 0);
        for plane_ctx in self.above_nonzero_context.iter_mut() {
            plane_ctx.iter_mut().for_each(|v| *v = 0);
        }
    }

    /// `clear_left_context( )` (spec §7.4.2): also clears `LeftSegPredContext`.
    fn clear_left_context(&mut self) {
        self.left_partition_context = [0u8; 8];
        self.left_seg_pred_context = [0u8; 8];
        self.left_nonzero_context = [[0u8; 16]; 3];
    }

    /// `decode_tiles( sz )` (spec §6.4). `data` is the entire tile data following the
    /// uncompressed header, excluding the compressed header (the rest of the frame data).
    pub fn decode_tiles(&mut self, mut data: &[u8]) -> Result<(), TileError> {
        let tile_cols = 1u32 << self.tile_cols_log2;
        let tile_rows = 1u32 << self.tile_rows_log2;
        self.clear_above_context();

        for tile_row in 0..tile_rows {
            for tile_col in 0..tile_cols {
                let last_tile = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
                let tile_size = if last_tile {
                    data.len()
                } else {
                    if data.len() < 4 {
                        return Err(TileError::InvalidTileSize);
                    }
                    let (size_bytes, rest) = data.split_at(4);
                    let ts = u32::from_be_bytes(size_bytes.try_into().unwrap()) as usize;
                    data = rest;
                    ts
                };
                if data.len() < tile_size {
                    return Err(TileError::InvalidTileSize);
                }
                let (tile_bytes, rest) = data.split_at(tile_size);
                data = rest;

                self.mi_row_start = get_tile_offset(tile_row, self.mi_rows, self.tile_rows_log2);
                self.mi_row_end = get_tile_offset(tile_row + 1, self.mi_rows, self.tile_rows_log2);
                self.mi_col_start = get_tile_offset(tile_col, self.mi_cols, self.tile_cols_log2);
                self.mi_col_end = get_tile_offset(tile_col + 1, self.mi_cols, self.tile_cols_log2);

                let mut r = BoolDecoder::new(tile_bytes).map_err(TileError::BoolCoder)?;
                self.decode_tile(&mut r)?;
                r.exit_bool();
            }
        }
        Ok(())
    }

    /// `decode_tile( )` (spec §6.4.2).
    fn decode_tile(&mut self, r: &mut BoolDecoder) -> Result<(), TileError> {
        let mut row = self.mi_row_start;
        while row < self.mi_row_end {
            self.clear_left_context();
            let mut col = self.mi_col_start;
            while col < self.mi_col_end {
                self.decode_partition(r, row, col, BLOCK_64X64)?;
                col += 8;
            }
            row += 8;
        }
        Ok(())
    }

    /// `decode_partition( r, c, bsize )` (spec §6.4.3).
    fn decode_partition(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        bsize: u8,
    ) -> Result<(), TileError> {
        if row >= self.mi_rows || col >= self.mi_cols {
            return Ok(());
        }
        let num8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as u32;
        let half_block8x8 = num8x8 >> 1;
        let has_rows = (row + half_block8x8) < self.mi_rows;
        let has_cols = (col + half_block8x8) < self.mi_cols;

        let partition = self.read_partition(r, (row, col), bsize, num8x8, has_rows, has_cols);
        let subsize = SUBSIZE_LOOKUP[partition as usize][bsize as usize];
        if subsize == BLOCK_INVALID {
            return Err(TileError::InvalidPartition);
        }

        if subsize < BLOCK_8X8 || partition == PARTITION_NONE {
            self.decode_block(r, row, col, subsize)?;
        } else if partition == PARTITION_HORZ {
            self.decode_block(r, row, col, subsize)?;
            if has_rows {
                self.decode_block(r, row + half_block8x8, col, subsize)?;
            }
        } else if partition == PARTITION_VERT {
            self.decode_block(r, row, col, subsize)?;
            if has_cols {
                self.decode_block(r, row, col + half_block8x8, subsize)?;
            }
        } else {
            debug_assert_eq!(partition, PARTITION_SPLIT);
            self.decode_partition(r, row, col, subsize)?;
            self.decode_partition(r, row, col + half_block8x8, subsize)?;
            self.decode_partition(r, row + half_block8x8, col, subsize)?;
            self.decode_partition(r, row + half_block8x8, col + half_block8x8, subsize)?;
        }

        if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
            let above_val = 15u8 >> B_WIDTH_LOG2_LOOKUP[subsize as usize];
            let left_val = 15u8 >> B_HEIGHT_LOG2_LOOKUP[subsize as usize];
            for i in 0..num8x8 {
                self.above_partition_context[(col + i) as usize] = above_val;
                self.left_partition_context[((row + i) % 8) as usize] = left_val;
            }
        }
        Ok(())
    }

    /// Reads the `partition` syntax element (spec §9.3.1 / §9.3.2).
    ///
    /// The spec §9.3.2 text states "use kf_partition_probs when FrameIsIntra == 0", but this
    /// is a known erratum. Since `compressed_header()` (spec §6.3) only reads
    /// `partition_probs` when `FrameIsIntra == 0`, implementing the text literally would mean
    /// referencing `partition_probs` while it is uninitialized and never updated for
    /// FrameIsIntra == 1 (key frames), which also contradicts the intent of the `kf_` prefix
    /// (a fixed table specific to key frames). This implementation corrects the wording: for
    /// `FrameIsIntra == 1` (key frames / intra-only frames) it uses the fixed table
    /// `KF_PARTITION_PROBS`, and for `FrameIsIntra == 0` (inter frames) it uses
    /// `partition_probs` as updated by `compressed_header()` (`self.probs.partition_probs`).
    fn read_partition(
        &mut self,
        r: &mut BoolDecoder,
        pos: (u32, u32),
        bsize: u8,
        num8x8: u32,
        has_rows: bool,
        has_cols: bool,
    ) -> u8 {
        let (row, col) = pos;
        let bsl = MI_WIDTH_LOG2_LOOKUP[bsize as usize] as u32;
        let boffset = MI_WIDTH_LOG2_LOOKUP[BLOCK_64X64 as usize] as u32 - bsl;
        let mut above = 0u32;
        let mut left = 0u32;
        for i in 0..num8x8 {
            above |= self.above_partition_context[(col + i) as usize] as u32;
            left |= self.left_partition_context[((row + i) % 8) as usize] as u32;
        }
        let above_bit = (above & (1 << boffset)) > 0;
        let left_bit = (left & (1 << boffset)) > 0;
        let ctx = (bsl * 4 + (left_bit as u32) * 2 + (above_bit as u32)) as usize;
        let probs = if self.frame_is_intra {
            &KF_PARTITION_PROBS[ctx]
        } else {
            &self.probs.partition_probs[ctx]
        };

        // Spec §9.3: `partition` is always a type T syntax element, and even when no bit is
        // read due to hasRows/hasCols (the tree selection process returns a value directly),
        // the counting process from spec §9.3.4 is always invoked (`counts_partition[ctx][syntax]`).
        let partition = if has_rows && has_cols {
            r.read_tree(&PARTITION_TREE, |node| probs[node]) as u8
        } else if has_cols {
            // cols_partition_tree: node2 is fixed at 1.
            if r.read_bool(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            // rows_partition_tree: node2 is fixed at 2.
            if r.read_bool(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if !self.frame_is_intra {
            self.counts.partition[ctx][partition as usize] += 1;
        }
        partition
    }

    /// `decode_block( r, c, subsize )` (spec §6.4.4).
    fn decode_block(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        subsize: u8,
    ) -> Result<(), TileError> {
        let avail_u = row > 0;
        let avail_l = col > self.mi_col_start;

        let mut info = if self.frame_is_intra {
            self.intra_frame_mode_info(r, row, col, subsize, avail_u, avail_l)?
        } else {
            self.inter_frame_mode_info(r, row, col, subsize, avail_u, avail_l)?
        };
        let is_inter = info.ref_frame[0] != INTRA_FRAME;
        let eob_total = self.residual(r, row, col, &info, avail_u, avail_l, is_inter);
        // Spec §6.4.4: when is_inter && subsize >= BLOCK_8X8 && EobTotal == 0, retroactively
        // set skip to 1 (since it turns out there was actually no residual at all).
        if is_inter && subsize >= BLOCK_8X8 && eob_total == 0 {
            info.skip = true;
        }

        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[subsize as usize] as u32;
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[subsize as usize] as u32;
        for y in 0..bh {
            for x in 0..bw {
                *self.mi_grid.get_mut(row + y, col + x) = info;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{
        ColorConfig, FrameType, LoopFilterParams, NewFrameHeader, QuantizationParams,
    };
    use crate::prob_tables::{
        BLOCK_4X4, BLOCK_64X64 as B64, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, ONLY_4X4,
        SINGLE_REFERENCE,
    };
    use crate::test_support::BoolEncoder;

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

    fn default_compressed_header() -> CompressedHeader {
        CompressedHeader {
            tx_mode: ONLY_4X4,
            probs: Arc::new(CompressedHeaderProbs::default()),
            reference_mode: SINGLE_REFERENCE,
            comp_fixed_ref: 0,
            comp_var_ref: [0, 0],
        }
    }

    #[test]
    fn get_tile_offset_matches_spec_formula() {
        // MiCols=1, tileSzLog2=0 -> only 1 tile, so offset is 0 and mis.
        assert_eq!(get_tile_offset(0, 1, 0), 0);
        assert_eq!(get_tile_offset(1, 1, 0), 1);
    }

    #[test]
    fn single_skip_block_decodes_without_residual_error() {
        // A frame consisting of a single superblock of 8x8 (1 MI).
        // partition (BLOCK_64X64, hasRows=false, hasCols=false) -> SPLIT with no bit read
        // partition (BLOCK_32X32, hasRows=false, hasCols=false) -> SPLIT
        // partition (BLOCK_16X16, hasRows=false, hasCols=false) -> SPLIT
        // partition (BLOCK_8X8, hasRows=false, hasCols=false) -> SPLIT, but
        //   num8x8=1 so half_block8x8=0, and hasRows/hasCols depend on the check.
        // Since MiCols=MiRows=1, half_block8x8 is always 0, so
        // hasRows = (r+0) < 1 = true (r=0), hasCols = true. So from the top level,
        // has_rows=has_cols=true and the whole tree must be read.
        let header = minimal_header(8, 8);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);

        let mut enc = BoolEncoder::new();
        // BLOCK_64X64: has_rows=true, has_cols=true (MiRows=MiCols=1, half=32>>... actually
        // num8x8=8, half=4, (0+4)<1 is false so hasRows=hasCols=false -> SPLIT with no bit read.
        // BLOCK_32X32: num8x8=4, half=2, (0+2)<1 false -> hasRows=hasCols=false -> SPLIT (no bit)
        // BLOCK_16X16: num8x8=2, half=1, (0+1)<1 false -> hasRows=hasCols=false -> SPLIT (no bit)
        // BLOCK_8X8: num8x8=1, half=0, (0+0)<1 true -> hasRows=hasCols=true -> read partition_tree
        let ctx = 3 * 4; // bsl(BLOCK_8X8)=0; the actual ctx depends on the above/left context. Left to the closure below.
        let _ = ctx;
        // Choose partition = PARTITION_NONE: bit=0 at the first branch of partition_tree.
        enc.write_bool(false, KF_PARTITION_PROBS[0][0]);
        // intra_frame_mode_info: skip=1
        enc.write_bool(true, CompressedHeaderProbs::default().skip_prob[0]);
        // read_tx_size: tx_mode=ONLY_4X4, so the tree is not read even with allowSelect.
        // default_intra_mode (MiSize=BLOCK_8X8 >= BLOCK_8X8): choose DC_PRED (bit=0 at the first tree branch)
        enc.write_bool(
            false,
            KF_Y_MODE_PROBS[DC_PRED as usize][DC_PRED as usize][0],
        );
        // default_uv_mode: DC_PRED
        enc.write_bool(false, KF_UV_MODE_PROBS[DC_PRED as usize][0]);
        let buf = enc.finish();

        decoder
            .decode_tiles(&buf)
            .expect("skip block should decode without error");

        let info = decoder.mi_grid().get(0, 0);
        assert!(info.skip);
        assert_eq!(info.mi_size, BLOCK_8X8);
        assert_eq!(info.y_mode, DC_PRED);
        assert_eq!(info.uv_mode, DC_PRED);
        assert_eq!(info.tx_size, TX_4X4);
        let _ = B64; // Explicitly use the BLOCK_64X64 alias import (avoid unused warning).
        let _ = BLOCK_4X4;
    }

    #[test]
    fn non_skip_block_with_all_zero_tokens_decodes_successfully() {
        // A non-skip 8x8 (1 MI) block. Since lossless, tx_size is always TX_4X4, and the Y
        // plane has 2x2=4 4x4 transform blocks, while U/V each have 1.
        // Setting the first more_coefs to false in each block (no coefficients) verifies the
        // full token decoding / inverse quantization / inverse transform / reconstruction
        // pipeline with a minimal configuration.
        let header = minimal_header(8, 8);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);

        let mut enc = BoolEncoder::new();
        enc.write_bool(false, KF_PARTITION_PROBS[0][0]); // PARTITION_NONE
        enc.write_bool(false, CompressedHeaderProbs::default().skip_prob[0]); // skip = 0
        enc.write_bool(
            false,
            KF_Y_MODE_PROBS[DC_PRED as usize][DC_PRED as usize][0],
        ); // DC_PRED
        enc.write_bool(false, KF_UV_MODE_PROBS[DC_PRED as usize][0]); // DC_PRED

        // DEFAULT_COEF_PROBS[TX_4X4][plane>0][is_inter=0][band=0][ctx=0][0]
        let y_more_coefs_prob = CompressedHeaderProbs::default().coef_probs[0][0][0][0][0][0];
        let uv_more_coefs_prob = CompressedHeaderProbs::default().coef_probs[0][1][0][0][0][0];
        for _ in 0..4 {
            enc.write_bool(false, y_more_coefs_prob); // Y's 4 4x4 blocks: more_coefs=0
        }
        enc.write_bool(false, uv_more_coefs_prob); // U
        enc.write_bool(false, uv_more_coefs_prob); // V
        let buf = enc.finish();

        decoder
            .decode_tiles(&buf)
            .expect("all-zero residual block should decode without error");

        let info = decoder.mi_grid().get(0, 0);
        assert!(!info.skip);
        // Since all coefficients are zero, the reconstructed pixel value should remain the
        // predicted value (DC_PRED, and since no reference is available, the bit_depth
        // midpoint 128).
        assert_eq!(decoder.planes()[0].get(0, 0), 128);
    }

    #[test]
    fn invalid_tile_size_is_rejected() {
        let header = minimal_header(64, 64);
        // 64x64 -> MiCols=8, Sb64Cols=1, so there is still only 1 tile, but force
        // tile_cols_log2 to 1 to require a tile size field.
        let mut header = header;
        header.tile_cols_log2 = 1;
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, header.color_config.unwrap(), &compressed);

        // Less than 4 bytes, so not even the tile size field can be read.
        let buf = [0u8; 2];
        let err = decoder.decode_tiles(&buf).unwrap_err();
        assert_eq!(err, TileError::InvalidTileSize);
    }
}
