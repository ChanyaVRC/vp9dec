//! タイル・スーパーブロック・モード情報・係数（トークン）の復号
//! （仕様 6.4 節 "Decode tiles syntax"）。
//!
//! M3 でインターフレーム（`FrameIsIntra == 0`）に対応し、`mode_info()` は `frame_is_intra`
//! に応じて `intra_frame_mode_info()`/`inter_frame_mode_info()` を呼び分ける。
//! `inter_frame_mode_info()`（仕様 6.4.11〜6.4.20 節）は動きベクトル予測（仕様 6.5 節、
//! `find_mv_refs`/`find_best_ref_mvs`/`append_sub8x8_mvs`）を含めて実装しているが、
//! **動き補償・サブピクセル補間（仕様 8.5.2 節 `predict_inter`）自体は未実装** で、
//! [`predict_inter_stub`](crate::predict::predict_inter_stub) を呼ぶのみに留めている
//! （ビットストリーム読了には影響しない。仕様 7.4.15 節 NOTE を参照。M3 後半で実装する）。
//!
//! [`TileDecoder::residual`] が仕様 6.4.21 節 `residual()` の実装であり、プレーンごとに
//! イントラ予測（[`crate::predict::predict_intra`]）→ トークン復号
//! （[`TileDecoder::tokens_and_reconstruct`]、仕様 6.4.24〜6.4.26 節）→ 逆量子化・逆変換・
//! 再構成（仕様 8.6.2 節）を行い、結果を [`TileDecoder::planes`] のフレームバッファに書き込む。
//! ループフィルタ（仕様 8.8 節）は M2b で実装予定のため未適用（ブロックノイズが残り得る）。

use crate::bool_coder::{BoolCoderError, BoolDecoder};
use crate::compressed_header::{CompressedHeader, CompressedHeaderProbs};
use crate::framebuffer::Plane;
use crate::header::{self, NewFrameHeader};
use crate::mv::{
    add_mv_ref_list, clamp_mv_col, clamp_mv_row, scale_mv, use_mv_hp, Mv, MVREF_NEIGHBOURS,
    MV_BORDER, MV_PRED_BORDER, ZERO_MV,
};
use crate::predict::predict_intra as predict_intra_block;
use crate::prob_tables::{
    coefband_8x8plus, mode2txfm_map, pareto, ALTREF_FRAME, BLOCK_64X64, BLOCK_8X8, BLOCK_INVALID,
    B_HEIGHT_LOG2_LOOKUP, B_WIDTH_LOG2_LOOKUP, CAT_PROBS, COEFBAND_4X4, COMPOUND_REFERENCE,
    COUNTER_TO_CONTEXT, DCT_VAL_CATEGORY6, DC_PRED, DEFAULT_UV_MODE_PROBS, ENERGY_CLASS,
    EXTRA_BITS, GOLDEN_FRAME, IDX_N_COLUMN_TO_SUBBLOCK, INTERP_FILTER_TREE, INTER_MODE_TREE,
    INTRA_FRAME, INTRA_MODE_TREE, KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
    LAST_FRAME, MAX_TXSIZE_LOOKUP, MI_WIDTH_LOG2_LOOKUP, MODE_2_COUNTER, MV_CLASS_TREE, MV_FR_TREE,
    MV_JOINT_HNZVNZ, MV_JOINT_HNZVZ, MV_JOINT_HZVNZ, MV_JOINT_TREE, MV_REF_BLOCKS, NEARESTMV,
    NEARMV, NEWMV, NUM_4X4_BLOCKS_HIGH_LOOKUP, NUM_4X4_BLOCKS_WIDE_LOOKUP,
    NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP, PARTITION_HORZ, PARTITION_NONE,
    PARTITION_SPLIT, PARTITION_TREE, PARTITION_VERT, REFERENCE_MODE_SELECT, REF_NONE,
    SINGLE_REFERENCE, SIZE_GROUP_LOOKUP, SS_SIZE_LOOKUP, SUBSIZE_LOOKUP, SWITCHABLE, TX_16X16,
    TX_32X32, TX_4X4, TX_8X8, TX_MODE_SELECT, TX_MODE_TO_BIGGEST_TX_SIZE, TX_SIZE_16_TREE,
    TX_SIZE_32_TREE, TX_SIZE_8_TREE, ZEROMV, ZERO_TOKEN,
};
use crate::quant::{get_ac_quant, get_dc_quant};
use crate::scan::{get_scan, TxSize};
use crate::transform::{inverse_transform_block, TxType};

/// タイル・パーティション復号時に発生し得るエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileError {
    /// タイルデータの bool デコーダ初期化に失敗した。
    BoolCoder(BoolCoderError),
    /// タイルサイズフィールドがデータ長を超えるなど、タイル分割が不正。
    InvalidTileSize,
    /// `subsize_lookup` が `BLOCK_INVALID` を返した（ビットストリーム不整合）。
    InvalidPartition,
    /// `segmentation_enabled == true` のフレーム。詳細なセグメンテーションパラメータ
    /// （`segmentation_update_map` 等）は `src/header.rs` がまだ保持していないため、
    /// 現時点ではキーフレームのセグメンテーションはサポート対象外。
    SegmentationNotSupported,
}

/// 1 つの 8x8 mode info 単位が保持する情報（仕様 2.37 節 "Mode info"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiInfo {
    pub skip: bool,
    pub tx_size: u8,
    pub mi_size: u8,
    /// `y_mode`（`MiSize >= BLOCK_8X8` の場合は `sub_modes` すべてと同じ値）。イントラブロック
    /// では `DC_PRED`..`TM_PRED`（0..9）、インターブロックでは `NEARESTMV`..`NEWMV`（10..13）。
    pub y_mode: u8,
    /// イントラブロックのみ意味を持つ（インターブロックでは未使用、0）。
    pub uv_mode: u8,
    /// 8x8 単位内の 4 つの 4x4 サブブロックのイントラモード（`MiSize < BLOCK_8X8` の場合のみ
    /// 複数の異なる値を取り得る）。インターブロックでは未使用（`[DC_PRED; 4]`）。
    pub sub_modes: [u8; 4],
    pub segment_id: u8,
    /// `ref_frame[ 0..2 ]`（仕様 7.4.12 節）。イントラブロックでは `[INTRA_FRAME, REF_NONE]`。
    /// `ref_frame[0] != INTRA_FRAME` であることが「インターブロックである」ことの判定条件
    /// （`is_inter`）そのもの。
    pub ref_frame: [u8; 2],
    /// `Mvs[ refList ]`（代表 MV、`BlockMvs[ refList ][ 3 ]` と同じ、仕様 6.4.4 節）。
    /// 単位は 1/8 pel。イントラブロックでは `[[0, 0]; 2]`。
    pub mv: [[i32; 2]; 2],
    /// `SubMvs[ refList ][ block 0..4 ]`（仕様 6.4.4 節）。`find_mv_refs` の
    /// `get_sub_block_mv` が近傍ブロックのサブ MV を参照する際に使う。
    pub sub_mvs: [[[i32; 2]; 4]; 2],
    /// `interp_filter`（仕様 6.4.16 節）。イントラブロックでは未使用（0）。
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

/// フレーム全体の mode info を 8x8 単位で保持するグリッド。
///
/// 仕様の `Skips`/`TxSizes`/`MiSizes`/`YModes`/`SegmentIds`/`SubModes` 各配列をまとめて
/// 1 つの構造体にしたもの。サイズは `Sb64Cols*8 x Sb64Rows*8`
/// （フレーム端のスーパーブロックがフレーム外にはみ出す分も含めて確保する）。
#[derive(Debug, Clone)]
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

/// `get_tile_offset( tileNum, mis, tileSzLog2 )`（仕様 6.4.1 節）。
fn get_tile_offset(tile_num: u32, mis: u32, tile_sz_log2: u32) -> u32 {
    let sbs = (mis + 7) >> 3;
    let offset = ((tile_num * sbs) >> tile_sz_log2) << 3;
    offset.min(mis)
}

/// タイル・スーパーブロックを走査してモード情報を復号するデコーダ。
///
/// 1 フレーム分の状態（mode info グリッド、above/left パーティションコンテキスト）を保持する。
pub struct TileDecoder {
    tx_mode: u8,
    probs: CompressedHeaderProbs,
    segmentation_enabled: bool,
    mi_cols: u32,
    mi_rows: u32,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    mi_grid: MiGrid,
    /// `AbovePartitionContext`。フレーム幅全体（`Sb64Cols*8`）にわたって持続し、
    /// フレーム毎に一度だけ（`decode_tiles` の先頭で）クリアされる（仕様 7.4.1 節）。
    above_partition_context: Vec<u8>,
    /// `LeftPartitionContext`。本来はフレーム高さ全体のサイズを持つ配列だが、
    /// スーパーブロック行ごとにクリアされ、常にそのスーパーブロック行内の相対位置
    /// （絶対 mi 行番号 mod 8）でしかアクセスされないため、8 要素の配列として保持する。
    left_partition_context: [u8; 8],
    // 現在デコード中のタイルの範囲。
    mi_col_start: u32,
    mi_col_end: u32,
    mi_row_start: u32,
    mi_row_end: u32,

    // --- 係数復号・再構成に必要な追加状態（仕様 6.4.21〜6.4.26 節、8.5〜8.6 節）。---
    bit_depth: u8,
    subsampling_x: u32,
    subsampling_y: u32,
    lossless: bool,
    base_q_idx: u8,
    delta_q_y_dc: i32,
    delta_q_uv_dc: i32,
    delta_q_uv_ac: i32,
    /// `CurrFrame[ plane ]`。インデックス 0=Y, 1=U, 2=V。
    planes: [Plane; 3],
    /// `AboveNonzeroContext[ plane ]`。4x4 単位でフレーム幅全体を持続する
    /// （フレーム毎に一度だけクリア）。
    above_nonzero_context: [Vec<u8>; 3],
    /// `LeftNonzeroContext[ plane ]`。スーパーブロック行ごとにクリアされ、常にそのスーパー
    /// ブロック行内の相対位置（絶対 4x4 行番号 mod 16）でしかアクセスされないため、
    /// 16 要素の配列として保持する（64 画素 / 4 = 16）。
    left_nonzero_context: [[u8; 16]; 3],

    // --- インターフレーム（仕様 6.4.11〜6.4.20 節・6.5 節）に必要な追加状態。---
    /// `FrameIsIntra`。
    frame_is_intra: bool,
    /// `ref_frame_sign_bias[ 0..4 ]`（非圧縮ヘッダ由来）。
    ref_frame_sign_bias: [bool; 4],
    allow_high_precision_mv: bool,
    /// `interpolation_filter`（`SWITCHABLE` の場合はブロックごとに `interp_filter` を読む）。
    interpolation_filter: u8,
    /// `reference_mode`/`CompFixedRef`/`CompVarRef`（圧縮ヘッダ由来、仕様 6.3.12・6.3.18 節）。
    reference_mode: u8,
    comp_fixed_ref: u8,
    comp_var_ref: [u8; 2],
    /// `UsePrevFrameMvs`（仕様 7.2.6 節）。真の場合、`prev_mi_grid` から前フレームの
    /// `Mvs`/`RefFrames` を参照する。
    use_prev_frame_mvs: bool,
    /// 直前にデコードしたフレームの `MiGrid`（`PrevMvs`/`PrevRefFrames` 相当）。
    /// `use_prev_frame_mvs == false` の場合は参照されない（`None` でもよい）。
    prev_mi_grid: Option<MiGrid>,
}

impl TileDecoder {
    /// 非圧縮ヘッダと圧縮ヘッダから `TileDecoder` を構築する。
    ///
    /// # Panics
    /// `header.color_config.bit_depth != 8` の場合にパニックする。8bit 以外
    /// （10bit/12bit）のフレームバッファは [`Plane`] が `u8` 固定であるため表現できない。
    /// 呼び出し側（[`crate::decode_keyframe`]）は `TileDecoder::new` を呼ぶ前に
    /// `bit_depth` を検査すること。`use_prev_frame_mvs`/`prev_mi_grid` は常に
    /// `false`/`None`（キーフレーム・M2 互換の単純なコンストラクタ）。
    pub fn new(header: &NewFrameHeader, compressed: &CompressedHeader) -> Self {
        Self::new_with_prev(header, compressed, false, None)
    }

    /// [`TileDecoder::new`] のインターフレーム対応版。`use_prev_frame_mvs`/`prev_mi_grid`
    /// は仕様 7.2.6 節の `UsePrevFrameMvs` とそれが参照する前フレームの `Mvs`/`RefFrames`
    /// （`find_mv_refs` の `usePrev` 分岐、仕様 6.5.1 節）に対応する。
    pub fn new_with_prev(
        header: &NewFrameHeader,
        compressed: &CompressedHeader,
        use_prev_frame_mvs: bool,
        prev_mi_grid: Option<MiGrid>,
    ) -> Self {
        assert_eq!(
            header.color_config.bit_depth, 8,
            "TileDecoder は 8bit フレームのみサポートする"
        );
        let image_size = header::compute_image_size(header.width, header.height);
        let grid_cols = (image_size.sb64_cols * 8) as usize;
        let grid_rows = (image_size.sb64_rows * 8) as usize;
        let subsampling_x = header.color_config.subsampling_x as u32;
        let subsampling_y = header.color_config.subsampling_y as u32;

        // フレームバッファはスーパーブロック境界（Sb64Cols*64 / Sb64Rows*64）まで
        // 切り上げて確保する（src/framebuffer.rs のドキュメント参照）。
        let y_w = (image_size.sb64_cols * 64) as usize;
        let y_h = (image_size.sb64_rows * 64) as usize;
        let uv_w = y_w >> subsampling_x;
        let uv_h = y_h >> subsampling_y;
        let planes = [
            Plane::new(y_w, y_h),
            Plane::new(uv_w, uv_h),
            Plane::new(uv_w, uv_h),
        ];
        // AboveNonzeroContext は 4x4 単位（8x8 mi 単位の 2 倍）でフレーム幅全体を持続する。
        let above_nz_len = grid_cols * 2;
        let above_nonzero_context = [
            vec![0u8; above_nz_len],
            vec![0u8; above_nz_len],
            vec![0u8; above_nz_len],
        ];

        Self {
            tx_mode: compressed.tx_mode,
            probs: compressed.probs.clone(),
            segmentation_enabled: header.segmentation_enabled,
            mi_cols: image_size.mi_cols,
            mi_rows: image_size.mi_rows,
            tile_cols_log2: header.tile_cols_log2,
            tile_rows_log2: header.tile_rows_log2,
            mi_grid: MiGrid::new(grid_cols, grid_rows),
            above_partition_context: vec![0u8; grid_cols],
            left_partition_context: [0u8; 8],
            mi_col_start: 0,
            mi_col_end: image_size.mi_cols,
            mi_row_start: 0,
            mi_row_end: image_size.mi_rows,
            bit_depth: header.color_config.bit_depth,
            subsampling_x,
            subsampling_y,
            lossless: header.quantization.lossless,
            base_q_idx: header.quantization.base_q_idx,
            delta_q_y_dc: header.quantization.delta_q_y_dc,
            delta_q_uv_dc: header.quantization.delta_q_uv_dc,
            delta_q_uv_ac: header.quantization.delta_q_uv_ac,
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
        }
    }

    pub fn mi_grid(&self) -> &MiGrid {
        &self.mi_grid
    }

    /// デコード済みのプレーンバッファ（`CurrFrame`）への参照を返す。
    /// インデックス 0=Y, 1=U, 2=V。バッファはスーパーブロック境界まで切り上げたサイズを
    /// 持つため、呼び出し側で表示サイズにクロップする必要がある
    /// （[`crate::decode_keyframe`] が行う）。
    pub fn planes(&self) -> &[Plane; 3] {
        &self.planes
    }

    /// ループフィルタ（仕様 8.8 節、[`crate::loop_filter`]）をデコード済みの `planes` に
    /// 適用する。`decode_tiles` の直後、`planes()` で出力を読み出す前に呼ぶこと。
    ///
    /// `self.planes`（`&mut`）と `self.mi_grid`（`&`）を別々のフィールドとして直接借用する
    /// ことで、両方を同時に必要とする `loop_filter_frame` に安全に渡している。
    pub fn apply_loop_filter(&mut self, lf: &header::LoopFilterParams) {
        crate::loop_filter::loop_filter_frame(
            &mut self.planes,
            &self.mi_grid,
            self.mi_cols,
            self.mi_rows,
            self.subsampling_x,
            self.subsampling_y,
            lf,
        );
    }

    fn clear_above_context(&mut self) {
        self.above_partition_context.iter_mut().for_each(|v| *v = 0);
        for plane_ctx in self.above_nonzero_context.iter_mut() {
            plane_ctx.iter_mut().for_each(|v| *v = 0);
        }
    }

    fn clear_left_context(&mut self) {
        self.left_partition_context = [0u8; 8];
        self.left_nonzero_context = [[0u8; 16]; 3];
    }

    /// `decode_tiles( sz )`（仕様 6.4 節）。`data` は非圧縮ヘッダ直後、
    /// 圧縮ヘッダを除いたタイルデータ全体（フレームデータの残り全部）。
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

    /// `decode_tile( )`（仕様 6.4.2 節）。
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

    /// `decode_partition( r, c, bsize )`（仕様 6.4.3 節）。
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

    /// `partition` シンタックス要素の読み取り（仕様 9.3.1 節・9.3.2 節）。
    ///
    /// 仕様 9.3.2 節の文面は「FrameIsIntra == 0 のとき kf_partition_probs を使う」と
    /// 記載しているが、これは既知の誤記である。`compressed_header()`
    /// （仕様 6.3 節）は `FrameIsIntra == 0` の場合にのみ `partition_probs` を読み込むため、
    /// 文面通りに実装すると FrameIsIntra == 1 (キーフレーム) で未初期化のまま更新されない
    /// `partition_probs` を参照することになり、`kf_` 接頭辞の意図（キーフレーム専用の固定表）
    /// とも矛盾する。本実装では文意を修正し、`FrameIsIntra == 1`（キーフレーム・イントラ
    /// オンリーフレーム）では固定表 `KF_PARTITION_PROBS` を、`FrameIsIntra == 0`
    /// （インターフレーム）では `compressed_header()` で更新される `partition_probs`
    /// （`self.probs.partition_probs`）を使う。
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

        if has_rows && has_cols {
            r.read_tree(&PARTITION_TREE, |node| probs[node]) as u8
        } else if has_cols {
            // cols_partition_tree: node2 は 1 に固定される。
            if r.read_bool(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            // rows_partition_tree: node2 は 2 に固定される。
            if r.read_bool(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        }
    }

    /// `decode_block( r, c, subsize )`（仕様 6.4.4 節）。
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
        // 仕様 6.4.4 節: is_inter && subsize >= BLOCK_8X8 && EobTotal == 0 の場合、
        // skip を後付けで 1 にする（残差が実際には全く無かったことが判明したため）。
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

    /// `get_uv_tx_size( )`（仕様 6.4.22 節）。
    fn get_uv_tx_size(&self, mi_size: u8, tx_size: u8) -> u8 {
        if mi_size < BLOCK_8X8 {
            return TX_4X4;
        }
        let plane_sz = self.get_plane_block_size(mi_size, 1);
        tx_size.min(MAX_TXSIZE_LOOKUP[plane_sz as usize])
    }

    /// `get_plane_block_size( subsize, plane )`（仕様 6.4.23 節）。
    fn get_plane_block_size(&self, subsize: u8, plane: usize) -> u8 {
        let subx = if plane > 0 { self.subsampling_x } else { 0 } as usize;
        let suby = if plane > 0 { self.subsampling_y } else { 0 } as usize;
        SS_SIZE_LOOKUP[subsize as usize][subx][suby]
    }

    /// `residual( )`（仕様 6.4.21 節）。戻り値は `EobTotal`（仕様 6.4.4 節）。
    #[allow(clippy::too_many_arguments)]
    fn residual(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        info: &MiInfo,
        avail_u: bool,
        avail_l: bool,
        is_inter: bool,
    ) -> u32 {
        let bsize = if info.mi_size < BLOCK_8X8 {
            BLOCK_8X8
        } else {
            info.mi_size
        };
        let mut eob_total = 0u32;

        for plane in 0..3usize {
            let tx_sz = if plane > 0 {
                self.get_uv_tx_size(info.mi_size, info.tx_size)
            } else {
                info.tx_size
            };
            let step = 1u32 << tx_sz;
            let plane_sz = self.get_plane_block_size(bsize, plane);
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize] as u32;
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize] as u32;
            let sub_x = if plane > 0 { self.subsampling_x } else { 0 };
            let sub_y = if plane > 0 { self.subsampling_y } else { 0 };
            let base_x = (col * 8) >> sub_x;
            let base_y = (row * 8) >> sub_y;
            let maxx = (self.mi_cols * 8) >> sub_x;
            let maxy = (self.mi_rows * 8) >> sub_y;
            // predict_intra に渡す maxX/maxY（仕様 8.5.1 節）はプレーン全体の座標での
            // クリップ境界であり、maxx/maxy(上記, residual のブロック内クリップ用) とは
            // 意味が異なる点に注意。
            let pred_max_x = ((self.mi_cols * 8) >> sub_x).saturating_sub(1) as usize;
            let pred_max_y = ((self.mi_rows * 8) >> sub_y).saturating_sub(1) as usize;

            if is_inter {
                // `predict_inter()`（仕様 8.5.2 節）: 動き補償・サブピクセル補間は M3 後半で
                // 実装する。ビットストリームの読み取り自体には影響しない（仕様 7.4.15 節 NOTE）
                // ため、ここではモード情報（`info.ref_frame`/`info.mv`/`info.interp_filter`）を
                // 記録済みであることを前提に、画素生成はスタブ呼び出しのみに留める。
                if info.mi_size < BLOCK_8X8 {
                    let mut y = 0u32;
                    while y < num4x4h {
                        let mut x = 0u32;
                        while x < num4x4w {
                            crate::predict::predict_inter_stub();
                            x += 1;
                        }
                        y += 1;
                    }
                } else {
                    crate::predict::predict_inter_stub();
                }
            }

            let mut block_idx = 0u32;
            let mut y = 0u32;
            while y < num4x4h {
                let mut x = 0u32;
                while x < num4x4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;

                    if start_x < maxx && start_y < maxy {
                        if !is_inter {
                            let mode = if plane > 0 {
                                info.uv_mode
                            } else if info.mi_size >= BLOCK_8X8 {
                                info.y_mode
                            } else {
                                info.sub_modes[block_idx as usize]
                            };
                            let have_left = avail_l || x > 0;
                            let have_above = avail_u || y > 0;
                            let not_on_right = x + step < num4x4w;
                            predict_intra_block(
                                &mut self.planes[plane],
                                start_x as usize,
                                start_y as usize,
                                have_left,
                                have_above,
                                not_on_right,
                                tx_sz,
                                mode,
                                pred_max_x,
                                pred_max_y,
                                self.bit_depth,
                            );
                        }

                        if !info.skip {
                            let tx_type = self.compute_tx_type(
                                plane,
                                tx_sz,
                                info.mi_size,
                                info.y_mode,
                                &info.sub_modes,
                                block_idx as usize,
                                is_inter,
                            );
                            nonzero = self.tokens_and_reconstruct(
                                r,
                                plane,
                                start_x as usize,
                                start_y as usize,
                                tx_sz,
                                tx_type,
                                is_inter,
                            );
                        }
                    }

                    eob_total += nonzero as u32;
                    for i in 0..step {
                        self.above_nonzero_context[plane][((start_x >> 2) + i) as usize] =
                            nonzero as u8;
                        let left_idx = (((start_y >> 2) + i) % 16) as usize;
                        self.left_nonzero_context[plane][left_idx] = nonzero as u8;
                    }
                    block_idx += 1;
                    x += step;
                }
                y += step;
            }
        }
        eob_total
    }

    /// `get_scan( )`（仕様 6.4.25 節）のうち `TxType` を決定する部分。
    #[allow(clippy::too_many_arguments)]
    fn compute_tx_type(
        &self,
        plane: usize,
        tx_sz: u8,
        mi_size: u8,
        y_mode: u8,
        sub_modes: &[u8; 4],
        block_idx: usize,
        is_inter: bool,
    ) -> TxType {
        if plane > 0 || tx_sz == TX_32X32 {
            TxType::DctDct
        } else if tx_sz == TX_4X4 {
            if self.lossless || is_inter {
                TxType::DctDct
            } else {
                let mode = if mi_size < BLOCK_8X8 {
                    sub_modes[block_idx]
                } else {
                    y_mode
                };
                mode2txfm_map(mode)
            }
        } else {
            // is_inter の場合 y_mode は NEARESTMV..NEWMV (10..13) を取り、
            // mode2txfm_map はこれらをすべて DctDct に写像する（仕様 10.2 節）。
            mode2txfm_map(y_mode)
        }
    }

    fn tx_sz_to_scan_size(tx_sz: u8) -> TxSize {
        match tx_sz {
            TX_4X4 => TxSize::Tx4x4,
            TX_8X8 => TxSize::Tx8x8,
            TX_16X16 => TxSize::Tx16x16,
            _ => TxSize::Tx32x32,
        }
    }

    /// `tokens( )`（仕様 6.4.24 節）+ `reconstruct( )`（仕様 8.6.2 節）。
    /// 戻り値は `nonzero`（仕様 6.4.24 節の `nonzero = c > 0`）。
    #[allow(clippy::too_many_arguments)]
    fn tokens_and_reconstruct(
        &mut self,
        r: &mut BoolDecoder,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: u8,
        tx_type: TxType,
        is_inter: bool,
    ) -> bool {
        let n = (tx_sz as u32) + 2;
        let n0 = 1usize << n;
        let seg_eob = n0 * n0;
        let scan = get_scan(Self::tx_sz_to_scan_size(tx_sz), tx_type);

        let plane_type = if plane > 0 { 1usize } else { 0usize };
        let sub_x = if plane > 0 { self.subsampling_x } else { 0 };
        let sub_y = if plane > 0 { self.subsampling_y } else { 0 };
        let max_x_ctx = (2 * self.mi_cols) >> sub_x;
        let max_y_ctx = (2 * self.mi_rows) >> sub_y;
        let numpts = 1u32 << tx_sz;
        let x4 = (start_x >> 2) as u32;
        let y4 = (start_y >> 2) as u32;

        let mut tokens = vec![0i32; seg_eob];
        let mut token_cache = [0u8; 1024];
        let mut check_eob = true;
        let mut c = 0usize;

        while c < seg_eob {
            let pos = scan[c] as usize;
            let band = if tx_sz == TX_4X4 {
                COEFBAND_4X4[c] as usize
            } else {
                coefband_8x8plus(c) as usize
            };

            // ctx の導出（仕様 9.3.2 節、more_coefs/token 共通）。
            let ctx = if c == 0 {
                let mut above = 0u32;
                let mut left = 0u32;
                for i in 0..numpts {
                    if x4 + i < max_x_ctx {
                        above |= self.above_nonzero_context[plane][(x4 + i) as usize] as u32;
                    }
                    if y4 + i < max_y_ctx {
                        left |= self.left_nonzero_context[plane][((y4 + i) % 16) as usize] as u32;
                    }
                }
                (above + left) as usize
            } else {
                let nn = 4usize << tx_sz;
                let i = pos / nn;
                let j = pos % nn;
                let (nb0, nb1) = if i > 0 && j > 0 {
                    let a = (i - 1) * nn + j;
                    let a2 = i * nn + j - 1;
                    match tx_type {
                        TxType::DctAdst => (a, a),
                        TxType::AdstDct => (a2, a2),
                        _ => (a, a2),
                    }
                } else if i > 0 {
                    let a = (i - 1) * nn + j;
                    (a, a)
                } else {
                    let a = i * nn + j - 1;
                    (a, a)
                };
                ((1 + token_cache[nb0] as u32 + token_cache[nb1] as u32) >> 1) as usize
            };

            let probs =
                &self.probs.coef_probs[tx_sz as usize][plane_type][is_inter as usize][band][ctx];

            if check_eob {
                let more_coefs = r.read_bool(probs[0]);
                if !more_coefs {
                    break;
                }
            }

            let token = r.read_tree(&crate::prob_tables::TOKEN_TREE, |node| {
                if node == 0 {
                    probs[1]
                } else if node == 1 {
                    probs[2]
                } else {
                    pareto(node, probs[2])
                }
            }) as u8;
            token_cache[pos] = ENERGY_CLASS[token as usize];

            if token == ZERO_TOKEN {
                tokens[pos] = 0;
                check_eob = false;
            } else {
                let coef = self.read_coef(r, token);
                let sign = r.read_literal(1) == 1;
                tokens[pos] = if sign { -coef } else { coef };
                check_eob = true;
            }

            c += 1;
        }

        let nonzero = c > 0;

        // 逆量子化 + 逆変換 + 再構成（仕様 8.6.2 節）。
        let dq_denom: i64 = if tx_sz == TX_32X32 { 2 } else { 1 };
        let qindex = self.base_q_idx;
        let ac_quant = get_ac_quant(self.bit_depth, qindex, plane, self.delta_q_uv_ac) as i64;
        let dc_quant = get_dc_quant(
            self.bit_depth,
            qindex,
            plane,
            self.delta_q_y_dc,
            self.delta_q_uv_dc,
        ) as i64;
        let mut dequant = vec![0i64; n0 * n0];
        for (idx, &t) in tokens.iter().enumerate() {
            dequant[idx] = (t as i64 * ac_quant) / dq_denom;
        }
        dequant[0] = (tokens[0] as i64 * dc_quant) / dq_denom;
        inverse_transform_block(&mut dequant, n, tx_type, self.lossless);

        let max_val = (1i64 << self.bit_depth) - 1;
        for i in 0..n0 {
            for j in 0..n0 {
                let old = self.planes[plane].get(start_x + j, start_y + i) as i64;
                let new_val = (old + dequant[i * n0 + j]).clamp(0, max_val);
                self.planes[plane].set(start_x + j, start_y + i, new_val as u8);
            }
        }

        nonzero
    }

    /// `read_coef( token )`（仕様 6.4.26 節）。
    fn read_coef(&self, r: &mut BoolDecoder, token: u8) -> i32 {
        let row = &EXTRA_BITS[token as usize];
        let cat = row[0] as usize;
        let num_extra = row[1] as u32;
        let mut coef = row[2] as i32;

        if token == DCT_VAL_CATEGORY6 {
            // BitDepth == 8 の場合、このループは 0 回実行される
            // （`for e in 0..(BitDepth-8)`）。10bit/12bit は M2 の対象外。
            for e in 0..(self.bit_depth.saturating_sub(8) as u32) {
                let high_bit = r.read_bool(255) as i32;
                coef += high_bit << (5 + self.bit_depth as u32 - e);
            }
        }

        for e in 0..num_extra {
            let coef_bit = r.read_bool(CAT_PROBS[cat][e as usize]) as i32;
            coef += coef_bit << (num_extra - 1 - e);
        }

        coef
    }

    /// `intra_segment_id( )`（仕様 6.4.7 節）。
    ///
    /// `segmentation_enabled == true` の場合、`segmentation_update_map` や
    /// `segmentation_tree_probs` が必要になるが、`src/header.rs` は M2 時点で
    /// `segmentation_enabled` の有無しか保持していない。そのため segmentation が有効な
    /// フレームは現時点ではサポート対象外としてエラーを返す
    /// （実データの `vp90-2-*.ivf` テストベクタでは `segmentation_enabled == false` であることを
    /// 確認済みで、少なくともこの範囲では影響がない）。
    fn intra_segment_id(&self) -> Result<u8, TileError> {
        if self.segmentation_enabled {
            Err(TileError::SegmentationNotSupported)
        } else {
            Ok(0)
        }
    }

    /// `read_skip( )`（仕様 6.4.8 節）。セグメンテーション未サポートのため
    /// `seg_feature_active( SEG_LVL_SKIP )` は常に false として扱う。
    fn read_skip(
        &self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        avail_u: bool,
        avail_l: bool,
    ) -> bool {
        let mut ctx = 0usize;
        if avail_u && self.mi_grid.get(row - 1, col).skip {
            ctx += 1;
        }
        if avail_l && self.mi_grid.get(row, col - 1).skip {
            ctx += 1;
        }
        r.read_bool(self.probs.skip_prob[ctx])
    }

    /// `read_tx_size( allowSelect )`（仕様 6.4.10 節）。
    fn read_tx_size(
        &self,
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
            let probs = &self.probs.tx_probs[max_tx_size as usize][ctx];
            match max_tx_size {
                TX_32X32 => r.read_tree(&TX_SIZE_32_TREE, |node| probs[node]) as u8,
                TX_16X16 => r.read_tree(&TX_SIZE_16_TREE, |node| probs[node]) as u8,
                _ => r.read_tree(&TX_SIZE_8_TREE, |node| probs[node]) as u8,
            }
        } else {
            max_tx_size.min(TX_MODE_TO_BIGGEST_TX_SIZE[self.tx_mode as usize])
        }
    }

    /// `intra_frame_mode_info( )`（仕様 6.4.6 節）。
    fn intra_frame_mode_info(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        mi_size: u8,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<MiInfo, TileError> {
        let segment_id = self.intra_segment_id()?;
        let skip = self.read_skip(r, row, col, avail_u, avail_l);
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
    // インターフレーム（`FrameIsIntra == 0`）のモード情報復号（仕様 6.4.11〜6.4.20 節）。
    // =========================================================================

    /// `inter_frame_mode_info( )`（仕様 6.4.11 節）。
    fn inter_frame_mode_info(
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

        // セグメンテーション未サポート（`intra_segment_id`/`inter_segment_id` は
        // segmentation_enabled == false の場合、どちらも常に 0 を返す点で等価）。
        let segment_id = self.intra_segment_id()?;
        let skip = self.read_skip(r, row, col, avail_u, avail_l);
        let is_inter = self.read_is_inter(r, &neighbors);
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

    /// `read_is_inter( )`（仕様 6.4.13 節）。`seg_feature_active( SEG_LVL_REF_FRAME )` は
    /// セグメンテーション未サポートのため常に false。
    fn read_is_inter(&self, r: &mut BoolDecoder, n: &NeighborRefInfo) -> bool {
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
        r.read_bool(self.probs.is_inter_prob[ctx])
    }

    /// `intra_block_mode_info( )`（仕様 6.4.15 節）。インターフレーム内のイントラブロック用。
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
                    // sub_intra_mode: ctx は常に 0（仕様 9.3.2 節）。
                    let mode = r
                        .read_tree(&INTRA_MODE_TREE, |node| self.probs.y_mode_probs[0][node])
                        as u8;
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
            DEFAULT_UV_MODE_PROBS[y_mode as usize][node]
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

    /// `read_ref_frames( )`（仕様 6.4.17 節）。`seg_feature_active( SEG_LVL_REF_FRAME )` は
    /// セグメンテーション未サポートのため常に false。
    fn read_ref_frames(&self, r: &mut BoolDecoder, n: &NeighborRefInfo) -> [u8; 2] {
        let comp_mode = if self.reference_mode == REFERENCE_MODE_SELECT {
            let ctx = self.comp_mode_ctx(n);
            (r.read_bool(self.probs.comp_mode_prob[ctx]) as u8) + SINGLE_REFERENCE
        } else {
            self.reference_mode
        };

        if comp_mode == COMPOUND_REFERENCE {
            let idx = self.ref_frame_sign_bias[self.comp_fixed_ref as usize] as usize;
            let ctx = self.comp_ref_ctx(n);
            let comp_ref = r.read_bool(self.probs.comp_ref_prob[ctx]) as usize;
            let mut ref_frame = [0u8; 2];
            ref_frame[idx] = self.comp_fixed_ref;
            ref_frame[1 - idx] = self.comp_var_ref[comp_ref];
            ref_frame
        } else {
            let ctx1 = self.single_ref_p1_ctx(n);
            let single_ref_p1 = r.read_bool(self.probs.single_ref_prob[ctx1][0]);
            if single_ref_p1 {
                let ctx2 = self.single_ref_p2_ctx(n);
                let single_ref_p2 = r.read_bool(self.probs.single_ref_prob[ctx2][1]);
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

    /// `comp_mode` の文脈導出（仕様 9.3.2 節）。
    fn comp_mode_ctx(&self, n: &NeighborRefInfo) -> usize {
        let fixed = self.comp_fixed_ref;
        if n.avail_u && n.avail_l {
            if n.above_single && n.left_single {
                ((n.above_ref_frame[0] == fixed) ^ (n.left_ref_frame[0] == fixed)) as usize
            } else if n.above_single {
                2 + (n.above_ref_frame[0] == fixed || n.above_intra) as usize
            } else if n.left_single {
                2 + (n.left_ref_frame[0] == fixed || n.left_intra) as usize
            } else {
                4
            }
        } else if n.avail_u {
            if n.above_single {
                (n.above_ref_frame[0] == fixed) as usize
            } else {
                3
            }
        } else if n.avail_l {
            if n.left_single {
                (n.left_ref_frame[0] == fixed) as usize
            } else {
                3
            }
        } else {
            1
        }
    }

    /// `comp_ref` の文脈導出（仕様 9.3.2 節）。
    fn comp_ref_ctx(&self, n: &NeighborRefInfo) -> usize {
        let fix_ref_idx = self.ref_frame_sign_bias[self.comp_fixed_ref as usize] as usize;
        let var_ref_idx = 1 - fix_ref_idx;
        let comp_var_ref = self.comp_var_ref;

        if n.avail_u && n.avail_l {
            if n.above_intra && n.left_intra {
                2
            } else if n.left_intra {
                if n.above_single {
                    1 + 2 * (n.above_ref_frame[0] != comp_var_ref[1]) as usize
                } else {
                    1 + 2 * (n.above_ref_frame[var_ref_idx] != comp_var_ref[1]) as usize
                }
            } else if n.above_intra {
                if n.left_single {
                    1 + 2 * (n.left_ref_frame[0] != comp_var_ref[1]) as usize
                } else {
                    1 + 2 * (n.left_ref_frame[var_ref_idx] != comp_var_ref[1]) as usize
                }
            } else {
                let vrfa = if n.above_single {
                    n.above_ref_frame[0]
                } else {
                    n.above_ref_frame[var_ref_idx]
                };
                let vrfl = if n.left_single {
                    n.left_ref_frame[0]
                } else {
                    n.left_ref_frame[var_ref_idx]
                };
                if vrfa == vrfl && comp_var_ref[1] == vrfa {
                    0
                } else if n.left_single && n.above_single {
                    if (vrfa == self.comp_fixed_ref && vrfl == comp_var_ref[0])
                        || (vrfl == self.comp_fixed_ref && vrfa == comp_var_ref[0])
                    {
                        4
                    } else if vrfa == vrfl {
                        3
                    } else {
                        1
                    }
                } else if n.left_single || n.above_single {
                    let vrfc = if n.left_single { vrfa } else { vrfl };
                    let rfs = if n.above_single { vrfa } else { vrfl };
                    if vrfc == comp_var_ref[1] && rfs != comp_var_ref[1] {
                        1
                    } else if rfs == comp_var_ref[1] && vrfc != comp_var_ref[1] {
                        2
                    } else {
                        4
                    }
                } else if vrfa == vrfl {
                    4
                } else {
                    2
                }
            }
        } else if n.avail_u {
            if n.above_intra {
                2
            } else if n.above_single {
                3 * (n.above_ref_frame[0] != comp_var_ref[1]) as usize
            } else {
                4 * (n.above_ref_frame[var_ref_idx] != comp_var_ref[1]) as usize
            }
        } else if n.avail_l {
            if n.left_intra {
                2
            } else if n.left_single {
                3 * (n.left_ref_frame[0] != comp_var_ref[1]) as usize
            } else {
                4 * (n.left_ref_frame[var_ref_idx] != comp_var_ref[1]) as usize
            }
        } else {
            2
        }
    }

    /// `single_ref_p1` の文脈導出（仕様 9.3.2 節）。
    fn single_ref_p1_ctx(&self, n: &NeighborRefInfo) -> usize {
        if n.avail_u && n.avail_l {
            if n.above_intra && n.left_intra {
                2
            } else if n.left_intra {
                if n.above_single {
                    4 * (n.above_ref_frame[0] == LAST_FRAME) as usize
                } else {
                    1 + (n.above_ref_frame[0] == LAST_FRAME || n.above_ref_frame[1] == LAST_FRAME)
                        as usize
                }
            } else if n.above_intra {
                if n.left_single {
                    4 * (n.left_ref_frame[0] == LAST_FRAME) as usize
                } else {
                    1 + (n.left_ref_frame[0] == LAST_FRAME || n.left_ref_frame[1] == LAST_FRAME)
                        as usize
                }
            } else if n.above_single && n.left_single {
                2 * (n.above_ref_frame[0] == LAST_FRAME) as usize
                    + 2 * (n.left_ref_frame[0] == LAST_FRAME) as usize
            } else if !n.above_single && !n.left_single {
                1 + (n.above_ref_frame[0] == LAST_FRAME
                    || n.above_ref_frame[1] == LAST_FRAME
                    || n.left_ref_frame[0] == LAST_FRAME
                    || n.left_ref_frame[1] == LAST_FRAME) as usize
            } else {
                let (rfs, crf1, crf2) = if n.above_single {
                    (
                        n.above_ref_frame[0],
                        n.left_ref_frame[0],
                        n.left_ref_frame[1],
                    )
                } else {
                    (
                        n.left_ref_frame[0],
                        n.above_ref_frame[0],
                        n.above_ref_frame[1],
                    )
                };
                if rfs == LAST_FRAME {
                    3 + (crf1 == LAST_FRAME || crf2 == LAST_FRAME) as usize
                } else {
                    (crf1 == LAST_FRAME || crf2 == LAST_FRAME) as usize
                }
            }
        } else if n.avail_u {
            if n.above_intra {
                2
            } else if n.above_single {
                4 * (n.above_ref_frame[0] == LAST_FRAME) as usize
            } else {
                1 + (n.above_ref_frame[0] == LAST_FRAME || n.above_ref_frame[1] == LAST_FRAME)
                    as usize
            }
        } else if n.avail_l {
            if n.left_intra {
                2
            } else if n.left_single {
                4 * (n.left_ref_frame[0] == LAST_FRAME) as usize
            } else {
                1 + (n.left_ref_frame[0] == LAST_FRAME || n.left_ref_frame[1] == LAST_FRAME)
                    as usize
            }
        } else {
            2
        }
    }

    /// `single_ref_p2` の文脈導出（仕様 9.3.2 節）。
    fn single_ref_p2_ctx(&self, n: &NeighborRefInfo) -> usize {
        if n.avail_u && n.avail_l {
            if n.above_intra && n.left_intra {
                2
            } else if n.left_intra {
                if n.above_single {
                    if n.above_ref_frame[0] == LAST_FRAME {
                        3
                    } else {
                        4 * (n.above_ref_frame[0] == GOLDEN_FRAME) as usize
                    }
                } else {
                    1 + 2
                        * (n.above_ref_frame[0] == GOLDEN_FRAME
                            || n.above_ref_frame[1] == GOLDEN_FRAME)
                            as usize
                }
            } else if n.above_intra {
                if n.left_single {
                    if n.left_ref_frame[0] == LAST_FRAME {
                        3
                    } else {
                        4 * (n.left_ref_frame[0] == GOLDEN_FRAME) as usize
                    }
                } else {
                    1 + 2
                        * (n.left_ref_frame[0] == GOLDEN_FRAME
                            || n.left_ref_frame[1] == GOLDEN_FRAME)
                            as usize
                }
            } else if n.above_single && n.left_single {
                if n.above_ref_frame[0] == LAST_FRAME && n.left_ref_frame[0] == LAST_FRAME {
                    3
                } else if n.above_ref_frame[0] == LAST_FRAME {
                    4 * (n.left_ref_frame[0] == GOLDEN_FRAME) as usize
                } else if n.left_ref_frame[0] == LAST_FRAME {
                    4 * (n.above_ref_frame[0] == GOLDEN_FRAME) as usize
                } else {
                    2 * (n.above_ref_frame[0] == GOLDEN_FRAME) as usize
                        + 2 * (n.left_ref_frame[0] == GOLDEN_FRAME) as usize
                }
            } else if !n.above_single && !n.left_single {
                if n.above_ref_frame[0] == n.left_ref_frame[0]
                    && n.above_ref_frame[1] == n.left_ref_frame[1]
                {
                    3 * (n.above_ref_frame[0] == GOLDEN_FRAME
                        || n.above_ref_frame[1] == GOLDEN_FRAME) as usize
                } else {
                    2
                }
            } else {
                let (rfs, crf1, crf2) = if n.above_single {
                    (
                        n.above_ref_frame[0],
                        n.left_ref_frame[0],
                        n.left_ref_frame[1],
                    )
                } else {
                    (
                        n.left_ref_frame[0],
                        n.above_ref_frame[0],
                        n.above_ref_frame[1],
                    )
                };
                if rfs == GOLDEN_FRAME {
                    3 + (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                } else if rfs == ALTREF_FRAME {
                    (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                } else {
                    1 + 2 * (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                }
            }
        } else if n.avail_u {
            if n.above_intra || (n.above_ref_frame[0] == LAST_FRAME && n.above_single) {
                2
            } else if n.above_single {
                4 * (n.above_ref_frame[0] == GOLDEN_FRAME) as usize
            } else {
                3 * (n.above_ref_frame[0] == GOLDEN_FRAME || n.above_ref_frame[1] == GOLDEN_FRAME)
                    as usize
            }
        } else if n.avail_l {
            if n.left_intra || (n.left_ref_frame[0] == LAST_FRAME && n.left_single) {
                2
            } else if n.left_single {
                4 * (n.left_ref_frame[0] == GOLDEN_FRAME) as usize
            } else {
                3 * (n.left_ref_frame[0] == GOLDEN_FRAME || n.left_ref_frame[1] == GOLDEN_FRAME)
                    as usize
            }
        } else {
            2
        }
    }

    /// `interp_filter` の文脈導出（仕様 9.3.2 節）。`3` は「両隣接ブロックの少なくとも一方が
    /// イントラ、またはフィルタが不一致」を表す番兵値。
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

    /// `inter_block_mode_info( )`（仕様 6.4.16 節）。
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
        let ref_frame = self.read_ref_frames(r, n);

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

        let mut y_mode = ZEROMV;
        if mi_size >= BLOCK_8X8 {
            let ctx = mode_context[ref_frame[0] as usize] as usize;
            let inter_mode = r.read_tree(&INTER_MODE_TREE, |node| {
                self.probs.inter_mode_probs[ctx][node]
            }) as u8;
            y_mode = NEARESTMV + inter_mode;
        }

        let interp_filter = if self.interpolation_filter == SWITCHABLE {
            let ctx = self.interp_filter_ctx(row, col, n);
            r.read_tree(&INTERP_FILTER_TREE, |node| {
                self.probs.interp_filter_probs[ctx][node]
            }) as u8
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

    /// `assign_mv( isCompound )`（仕様 6.4.18 節）。
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

    /// `read_mv( ref )`（仕様 6.4.19 節）。
    fn read_mv(&mut self, r: &mut BoolDecoder, best_mv: Mv) -> Mv {
        let use_hp = self.allow_high_precision_mv && use_mv_hp(best_mv);
        let mv_joint = r.read_tree(&MV_JOINT_TREE, |node| self.probs.mv_joint_probs[node]) as u8;
        let mut diff = ZERO_MV;
        if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
            diff[0] = self.read_mv_component(r, 0, use_hp);
        }
        if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
            diff[1] = self.read_mv_component(r, 1, use_hp);
        }
        [best_mv[0] + diff[0], best_mv[1] + diff[1]]
    }

    /// `read_mv_component( comp )`（仕様 6.4.20 節）。
    fn read_mv_component(&mut self, r: &mut BoolDecoder, comp: usize, use_hp: bool) -> i32 {
        let sign = r.read_bool(self.probs.mv_sign_prob[comp]);
        let mv_class =
            r.read_tree(&MV_CLASS_TREE, |node| self.probs.mv_class_probs[comp][node]) as usize;
        let mag: u32 = if mv_class == 0 {
            let class0_bit = r.read_bool(self.probs.mv_class0_bit_prob[comp]) as u32;
            let class0_fr = r.read_tree(&MV_FR_TREE, |node| {
                self.probs.mv_class0_fr_probs[comp][class0_bit as usize][node]
            }) as u32;
            let class0_hp = if use_hp {
                r.read_bool(self.probs.mv_class0_hp_prob[comp]) as u32
            } else {
                1
            };
            ((class0_bit << 3) | (class0_fr << 1) | class0_hp) + 1
        } else {
            let mut d: u32 = 0;
            for i in 0..mv_class {
                let bit = r.read_bool(self.probs.mv_bits_prob[comp][i]) as u32;
                d |= bit << i;
            }
            let mut mag = 2u32 << (mv_class + 2); // CLASS0_SIZE(2) << (mv_class+2)
            let fr = r.read_tree(&MV_FR_TREE, |node| self.probs.mv_fr_probs[comp][node]) as u32;
            let hp = if use_hp {
                r.read_bool(self.probs.mv_hp_prob[comp]) as u32
            } else {
                1
            };
            mag += ((d << 3) | (fr << 1) | hp) + 1;
            mag
        };
        if sign {
            -(mag as i32)
        } else {
            mag as i32
        }
    }

    /// `is_inside( candidateR, candidateC )`（仕様 6.5.2 節）。
    fn is_inside(&self, r: i32, c: i32) -> bool {
        r >= 0
            && (r as u32) < self.mi_rows
            && c >= self.mi_col_start as i32
            && c < self.mi_col_end as i32
    }

    /// `get_block_mv( candidateR, candidateC, refList, usePrev )`（仕様 6.5.10 節）。
    /// 戻り値は `(CandidateMv, CandidateFrame)`。
    fn get_block_mv(&self, row: u32, col: u32, ref_list: usize, use_prev: bool) -> (Mv, u8) {
        if use_prev {
            let grid = self
                .prev_mi_grid
                .as_ref()
                .expect("use_prev_frame_mvs が真の場合 prev_mi_grid は必ず Some");
            let info = grid.get(row, col);
            (info.mv[ref_list], info.ref_frame[ref_list])
        } else {
            let info = self.mi_grid.get(row, col);
            (info.mv[ref_list], info.ref_frame[ref_list])
        }
    }

    /// `if_same_ref_frame_add_mv( candidateR, candidateC, refFrame, usePrev )`（仕様 6.5.7 節）。
    fn if_same_ref_frame_add_mv(
        &self,
        row: u32,
        col: u32,
        ref_frame: u8,
        use_prev: bool,
        ref_list_mv: &mut [Mv; 2],
        ref_mv_count: &mut usize,
    ) {
        for j in 0..2 {
            let (cand_mv, cand_frame) = self.get_block_mv(row, col, j, use_prev);
            if cand_frame == ref_frame {
                add_mv_ref_list(ref_list_mv, ref_mv_count, cand_mv);
                return;
            }
        }
    }

    /// `if_diff_ref_frame_add_mv( candidateR, candidateC, refFrame, usePrev )`（仕様 6.5.8 節）。
    fn if_diff_ref_frame_add_mv(
        &self,
        row: u32,
        col: u32,
        ref_frame: u8,
        use_prev: bool,
        ref_list_mv: &mut [Mv; 2],
        ref_mv_count: &mut usize,
    ) {
        let (mv0, frame0) = self.get_block_mv(row, col, 0, use_prev);
        let (mv1, frame1) = self.get_block_mv(row, col, 1, use_prev);
        let mvs_same = mv0 == mv1;
        if frame0 > INTRA_FRAME && frame0 != ref_frame {
            let scaled = scale_mv(mv0, frame0, ref_frame, &self.ref_frame_sign_bias);
            add_mv_ref_list(ref_list_mv, ref_mv_count, scaled);
        }
        if frame1 > INTRA_FRAME && frame1 != ref_frame && !mvs_same {
            let scaled = scale_mv(mv1, frame1, ref_frame, &self.ref_frame_sign_bias);
            add_mv_ref_list(ref_list_mv, ref_mv_count, scaled);
        }
    }

    /// `find_mv_refs( refFrame, block )`（仕様 6.5.1 節）。戻り値は `(RefListMv, ModeContext)`。
    fn find_mv_refs(
        &self,
        row: u32,
        col: u32,
        mi_size: u8,
        ref_frame: u8,
        block: i32,
    ) -> ([Mv; 2], u8) {
        let mut ref_list_mv = [ZERO_MV; 2];
        let mut ref_mv_count = 0usize;
        let mut different_ref_found = false;
        let mut context_counter: u32 = 0;

        let search = &MV_REF_BLOCKS[mi_size as usize];

        for &(dr, dc) in search.iter().take(2) {
            let cr = row as i32 + dr;
            let cc = col as i32 + dc;
            if self.is_inside(cr, cc) {
                different_ref_found = true;
                let cand = self.mi_grid.get(cr as u32, cc as u32);
                context_counter += MODE_2_COUNTER[cand.y_mode as usize] as u32;
                for j in 0..2 {
                    if cand.ref_frame[j] == ref_frame {
                        let idx = if block >= 0 {
                            IDX_N_COLUMN_TO_SUBBLOCK[block as usize][(dc == 0) as usize] as usize
                        } else {
                            3
                        };
                        add_mv_ref_list(&mut ref_list_mv, &mut ref_mv_count, cand.sub_mvs[j][idx]);
                        break;
                    }
                }
            }
        }

        for &(dr, dc) in search.iter().skip(2).take(MVREF_NEIGHBOURS - 2) {
            let cr = row as i32 + dr;
            let cc = col as i32 + dc;
            if self.is_inside(cr, cc) {
                different_ref_found = true;
                self.if_same_ref_frame_add_mv(
                    cr as u32,
                    cc as u32,
                    ref_frame,
                    false,
                    &mut ref_list_mv,
                    &mut ref_mv_count,
                );
            }
        }

        if self.use_prev_frame_mvs {
            self.if_same_ref_frame_add_mv(
                row,
                col,
                ref_frame,
                true,
                &mut ref_list_mv,
                &mut ref_mv_count,
            );
        }

        if different_ref_found {
            for &(dr, dc) in search.iter().take(MVREF_NEIGHBOURS) {
                let cr = row as i32 + dr;
                let cc = col as i32 + dc;
                if self.is_inside(cr, cc) {
                    self.if_diff_ref_frame_add_mv(
                        cr as u32,
                        cc as u32,
                        ref_frame,
                        false,
                        &mut ref_list_mv,
                        &mut ref_mv_count,
                    );
                }
            }
        }

        if self.use_prev_frame_mvs {
            self.if_diff_ref_frame_add_mv(
                row,
                col,
                ref_frame,
                true,
                &mut ref_list_mv,
                &mut ref_mv_count,
            );
        }

        let mode_context = COUNTER_TO_CONTEXT[context_counter.min(18) as usize];

        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
        for mv in ref_list_mv.iter_mut() {
            mv[0] = clamp_mv_row(mv[0], MV_BORDER, row, bh, self.mi_rows);
            mv[1] = clamp_mv_col(mv[1], MV_BORDER, col, bw, self.mi_cols);
        }

        (ref_list_mv, mode_context)
    }

    /// `find_best_ref_mvs( refList )`（仕様 6.5.12 節）。戻り値は `(NearestMv, NearMv, BestMv)`。
    fn find_best_ref_mvs(
        &self,
        row: u32,
        col: u32,
        mi_size: u8,
        ref_list_mv: [Mv; 2],
    ) -> (Mv, Mv, Mv) {
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
        let mut out = ref_list_mv;
        for mv in out.iter_mut() {
            let mut delta_row = mv[0];
            let mut delta_col = mv[1];
            if !self.allow_high_precision_mv || !use_mv_hp(*mv) {
                if delta_row & 1 != 0 {
                    delta_row += if delta_row > 0 { -1 } else { 1 };
                }
                if delta_col & 1 != 0 {
                    delta_col += if delta_col > 0 { -1 } else { 1 };
                }
            }
            mv[0] = clamp_mv_row(delta_row, MV_PRED_BORDER, row, bh, self.mi_rows);
            mv[1] = clamp_mv_col(delta_col, MV_PRED_BORDER, col, bw, self.mi_cols);
        }
        (out[0], out[1], out[0])
    }

    /// `append_sub8x8_mvs( block, refList )`（仕様 6.5.14 節）。戻り値は `(NearestMv, NearMv)`。
    #[allow(clippy::too_many_arguments)]
    fn append_sub8x8_mvs(
        &self,
        row: u32,
        col: u32,
        mi_size: u8,
        block: i32,
        ref_frame: u8,
        ref_list: usize,
        block_mvs: &[[Mv; 4]; 2],
    ) -> (Mv, Mv) {
        let (ref_list_mv, _ctx) = self.find_mv_refs(row, col, mi_size, ref_frame, block);
        let mut sub8x8: Vec<Mv> = Vec::with_capacity(2);

        if block == 0 {
            sub8x8.push(ref_list_mv[0]);
            sub8x8.push(ref_list_mv[1]);
        } else if block <= 2 {
            sub8x8.push(block_mvs[ref_list][0]);
        } else {
            sub8x8.push(block_mvs[ref_list][2]);
            for &idx in &[1usize, 0] {
                if sub8x8.len() >= 2 {
                    break;
                }
                if block_mvs[ref_list][idx] != sub8x8[0] {
                    sub8x8.push(block_mvs[ref_list][idx]);
                }
            }
        }
        for &cand in ref_list_mv.iter().take(2) {
            if sub8x8.len() >= 2 {
                break;
            }
            if cand != sub8x8[0] {
                sub8x8.push(cand);
            }
        }
        if sub8x8.len() < 2 {
            sub8x8.push(ZERO_MV);
        }
        (sub8x8[0], sub8x8[1])
    }
}

/// `inter_frame_mode_info( )`（仕様 6.4.11 節）が導出する近傍情報
/// （`LeftRefFrame`/`AboveRefFrame`/`LeftIntra`/`AboveIntra`/`LeftSingle`/`AboveSingle`）。
struct NeighborRefInfo {
    avail_u: bool,
    avail_l: bool,
    left_ref_frame: [u8; 2],
    above_ref_frame: [u8; 2],
    left_intra: bool,
    above_intra: bool,
    left_single: bool,
    above_single: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::test_support::BoolEncoder;
    use crate::header::{
        ColorConfig, FrameType, LoopFilterParams, NewFrameHeader, QuantizationParams,
    };
    use crate::prob_tables::{BLOCK_4X4, BLOCK_64X64 as B64, ONLY_4X4};

    /// テスト用に最小限の `NewFrameHeader` を組み立てる。8x8 (1 MI, 1 SB) のキーフレーム。
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
            color_config: ColorConfig {
                bit_depth: 8,
                color_space: 0,
                color_range: false,
                subsampling_x: 1,
                subsampling_y: 1,
            },
            width,
            height,
            render_width: width,
            render_height: height,
            refresh_frame_flags: 0xFF,
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
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
            segmentation_enabled: false,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            header_size_in_bytes: 0,
        }
    }

    fn default_compressed_header() -> CompressedHeader {
        CompressedHeader {
            tx_mode: ONLY_4X4,
            probs: CompressedHeaderProbs::default(),
            reference_mode: SINGLE_REFERENCE,
            comp_fixed_ref: 0,
            comp_var_ref: [0, 0],
        }
    }

    #[test]
    fn get_tile_offset_matches_spec_formula() {
        // MiCols=1, tileSzLog2=0 -> 1 タイルのみで offset は 0 と mis になる。
        assert_eq!(get_tile_offset(0, 1, 0), 0);
        assert_eq!(get_tile_offset(1, 1, 0), 1);
    }

    #[test]
    fn single_skip_block_decodes_without_residual_error() {
        // 8x8 (1 MI) の 1 スーパーブロックだけのフレーム。
        // partition (BLOCK_64X64, hasRows=false, hasCols=false) -> ビットを読まず SPLIT
        // partition (BLOCK_32X32, hasRows=false, hasCols=false) -> SPLIT
        // partition (BLOCK_16X16, hasRows=false, hasCols=false) -> SPLIT
        // partition (BLOCK_8X8, hasRows=false, hasCols=false) -> SPLIT だが
        //   num8x8=1 なので half_block8x8=0 となり hasRows/hasCols は判定次第。
        // MiCols=MiRows=1 なので half_block8x8 は常に 0 のため
        // hasRows = (r+0) < 1 = true (r=0), hasCols = true。よって最上位から
        // has_rows=has_cols=true でツリー全体を読む必要がある。
        let header = minimal_header(8, 8);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);

        let mut enc = BoolEncoder::new();
        // BLOCK_64X64: has_rows=true, has_cols=true (MiRows=MiCols=1, half=32>>... 実際は
        // num8x8=8, half=4, (0+4)<1 は false なので hasRows=hasCols=false -> ビットなしで SPLIT。
        // BLOCK_32X32: num8x8=4, half=2, (0+2)<1 false -> hasRows=hasCols=false -> SPLIT (ビットなし)
        // BLOCK_16X16: num8x8=2, half=1, (0+1)<1 false -> hasRows=hasCols=false -> SPLIT (ビットなし)
        // BLOCK_8X8: num8x8=1, half=0, (0+0)<1 true -> hasRows=hasCols=true -> partition_tree を読む
        let ctx = 3 * 4; // bsl(BLOCK_8X8)=0 の実際の ctx は above/left context 次第。後述のクロージャに委ねる。
        let _ = ctx;
        // partition = PARTITION_NONE を選択する: partition_tree の最初の分岐で bit=0。
        enc.write_bool(false, KF_PARTITION_PROBS[0][0]);
        // intra_frame_mode_info: skip=1
        enc.write_bool(true, CompressedHeaderProbs::default().skip_prob[0]);
        // read_tx_size: tx_mode=ONLY_4X4 なので allowSelect でもツリーは読まれない。
        // default_intra_mode (MiSize=BLOCK_8X8 >= BLOCK_8X8): DC_PRED を選択(木の最初の分岐 bit=0)
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
        let _ = B64; // BLOCK_64X64 の別名インポートを使用していることの明示 (未使用警告防止)。
        let _ = BLOCK_4X4;
    }

    #[test]
    fn non_skip_block_with_all_zero_tokens_decodes_successfully() {
        // 8x8 (1 MI) の非 skip ブロック。lossless なので tx_size は常に TX_4X4 で、
        // Y プレーンは 2x2=4 個、U/V プレーンは 1 個ずつの 4x4 変換ブロックを持つ。
        // 各ブロックで最初の more_coefs を false にする（係数なし）ことで、
        // トークン復号・逆量子化・逆変換・再構成の一連の処理を最小構成で検証する。
        let header = minimal_header(8, 8);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);

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
            enc.write_bool(false, y_more_coefs_prob); // Y の 4 個の 4x4 ブロック: more_coefs=0
        }
        enc.write_bool(false, uv_more_coefs_prob); // U
        enc.write_bool(false, uv_more_coefs_prob); // V
        let buf = enc.finish();

        decoder
            .decode_tiles(&buf)
            .expect("all-zero residual block should decode without error");

        let info = decoder.mi_grid().get(0, 0);
        assert!(!info.skip);
        // 全係数ゼロなので、再構成後のピクセル値は予測値（DC_PRED, 参照不可なので
        // bit_depth の中間値 128）のままのはず。
        assert_eq!(decoder.planes()[0].get(0, 0), 128);
    }

    #[test]
    fn segmentation_enabled_is_rejected() {
        let mut header = minimal_header(8, 8);
        header.segmentation_enabled = true;
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);

        // 中身がどうであれ intra_segment_id() の時点でエラーになる。
        let buf = [0u8; 4];
        let err = decoder.decode_tiles(&buf).unwrap_err();
        assert_eq!(err, TileError::SegmentationNotSupported);
    }

    #[test]
    fn invalid_tile_size_is_rejected() {
        let header = minimal_header(64, 64);
        // 64x64 -> MiCols=8, Sb64Cols=1 なのでタイルは 1 つのままだが、
        // tile_cols_log2 を強制的に 1 にしてタイルサイズフィールドを要求させる。
        let mut header = header;
        header.tile_cols_log2 = 1;
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);

        // 4 バイト未満なのでタイルサイズフィールドすら読めない。
        let buf = [0u8; 2];
        let err = decoder.decode_tiles(&buf).unwrap_err();
        assert_eq!(err, TileError::InvalidTileSize);
    }

    // =========================================================================
    // MV 復号（仕様 6.4.19〜6.4.20 節）の単体テスト。
    // `BoolEncoder`（test_support）で既知の MV をエンコードし、`TileDecoder::read_mv`/
    // `read_mv_component`（非公開メソッド）で復号して往復一致を確認する。
    // =========================================================================

    fn minimal_inter_header(width: u32, height: u32) -> NewFrameHeader {
        let mut h = minimal_header(width, height);
        h.frame_type = FrameType::NonKeyFrame;
        h.frame_is_intra = false;
        h.ref_frame_idx = [0, 1, 2];
        h
    }

    #[test]
    fn read_mv_component_class0_roundtrip() {
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);
        let probs = CompressedHeaderProbs::default();

        // mv_sign=0(正), mv_class=MV_CLASS_0, class0_bit=1, class0_fr=2, (use_hp=false のため
        // class0_hp は読まれず 1 とみなされる)。
        // mag = ((1<<3)|(2<<1)|1) + 1 = 13 + 1 = 14
        let mut enc = BoolEncoder::new();
        enc.write_bool(false, probs.mv_sign_prob[0]);
        enc.write_bool(false, probs.mv_class_probs[0][0]); // MV_CLASS_0 (tree leaf)
        enc.write_bool(true, probs.mv_class0_bit_prob[0]); // class0_bit = 1
                                                           // class0_fr = 2: MV_FR_TREE のビット列 [1,1,0]
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
        let mut decoder = TileDecoder::new(&header, &compressed);
        let probs = CompressedHeaderProbs::default();

        let mut enc = BoolEncoder::new();
        enc.write_bool(true, probs.mv_sign_prob[1]); // sign = 負
        enc.write_bool(false, probs.mv_class_probs[1][0]); // MV_CLASS_0
        enc.write_bool(false, probs.mv_class0_bit_prob[1]); // class0_bit = 0
                                                            // class0_fr = 0: ビット列 [0]
        enc.write_bool(false, probs.mv_class0_fr_probs[1][0][0]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        // mag = ((0<<3)|(0<<1)|1) + 1 = 2、符号が負なので -2。
        let mag = decoder.read_mv_component(&mut r, 1, false);
        assert_eq!(mag, -2);
    }

    #[test]
    fn read_mv_component_higher_class_roundtrip() {
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);
        let probs = CompressedHeaderProbs::default();

        // mv_class = MV_CLASS_1 (値 1): ツリー [0,2,-1,4,...] で bit0=1,bit1=0 -> leaf -1 (値1)。
        let mut enc = BoolEncoder::new();
        enc.write_bool(false, probs.mv_sign_prob[0]); // 正
        enc.write_bool(true, probs.mv_class_probs[0][0]);
        enc.write_bool(false, probs.mv_class_probs[0][1]);
        // mv_class=1 -> d のビットを 1 個読む (mv_bit)。d=1 とする。
        enc.write_bool(true, probs.mv_bits_prob[0][0]);
        // mv_fr = 1: ビット列 [1,0]
        enc.write_bool(true, probs.mv_fr_probs[0][0]);
        enc.write_bool(false, probs.mv_fr_probs[0][1]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        // mag = CLASS0_SIZE << (1+2) = 2<<3 = 16; d=1 (bit0=1), fr=1, hp(強制1)
        // mag += ((1<<3)|(1<<1)|1) + 1 = (8|2|1)+1 = 11+1 = 12 -> total 16+12 = 28
        let mag = decoder.read_mv_component(&mut r, 0, false);
        assert_eq!(mag, 28);
    }

    #[test]
    fn read_mv_full_roundtrip_with_best_mv_offset() {
        // read_mv(ref) = BestMv + diffMv。mv_joint = MV_JOINT_HNZVNZ (両成分とも非ゼロ)。
        let header = minimal_inter_header(64, 64);
        let compressed = default_compressed_header();
        let mut decoder = TileDecoder::new(&header, &compressed);
        let probs = CompressedHeaderProbs::default();
        let best_mv: Mv = [10, -20];

        let mut enc = BoolEncoder::new();
        // mv_joint tree: MV_JOINT_HNZVNZ(=3) のビット列 [1,1,1]
        enc.write_bool(true, probs.mv_joint_probs[0]);
        enc.write_bool(true, probs.mv_joint_probs[1]);
        enc.write_bool(true, probs.mv_joint_probs[2]);
        // comp 0 (row): mag=14 (前テストと同じ class0 パターン、符号正)
        enc.write_bool(false, probs.mv_sign_prob[0]);
        enc.write_bool(false, probs.mv_class_probs[0][0]);
        enc.write_bool(true, probs.mv_class0_bit_prob[0]);
        enc.write_bool(true, probs.mv_class0_fr_probs[0][1][0]);
        enc.write_bool(true, probs.mv_class0_fr_probs[0][1][1]);
        enc.write_bool(false, probs.mv_class0_fr_probs[0][1][2]);
        // comp 1 (col): mag=2、符号負 -> -2
        enc.write_bool(true, probs.mv_sign_prob[1]);
        enc.write_bool(false, probs.mv_class_probs[1][0]);
        enc.write_bool(false, probs.mv_class0_bit_prob[1]);
        enc.write_bool(false, probs.mv_class0_fr_probs[1][0][0]);
        let buf = enc.finish();

        let mut r = BoolDecoder::new(&buf).expect("valid bitstream");
        // allow_high_precision_mv=false なので use_hp は常に false（use_mv_hp の結果に関わらず）。
        let mv = decoder.read_mv(&mut r, best_mv);
        assert_eq!(mv, [10 + 14, -20 + (-2)]);
    }
}
