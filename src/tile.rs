//! タイル・スーパーブロック・モード情報・係数（トークン）の復号
//! （仕様 6.4 節 "Decode tiles syntax"）。
//!
//! M2 の対象はキーフレーム（`FrameIsIntra == 1`）のイントラ復号のみである。そのため
//! `mode_info()` は `intra_frame_mode_info()` のみを実装し、`inter_frame_mode_info()` は
//! 実装しない（M3 で追加する）。
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
use crate::predict::predict_intra as predict_intra_block;
use crate::prob_tables::{
    coefband_8x8plus, mode2txfm_map, pareto, BLOCK_64X64, BLOCK_8X8, BLOCK_INVALID,
    B_HEIGHT_LOG2_LOOKUP, B_WIDTH_LOG2_LOOKUP, CAT_PROBS, COEFBAND_4X4, DCT_VAL_CATEGORY6, DC_PRED,
    ENERGY_CLASS, EXTRA_BITS, INTRA_MODE_TREE, KF_PARTITION_PROBS, KF_UV_MODE_PROBS,
    KF_Y_MODE_PROBS, MAX_TXSIZE_LOOKUP, MI_WIDTH_LOG2_LOOKUP, NUM_4X4_BLOCKS_HIGH_LOOKUP,
    NUM_4X4_BLOCKS_WIDE_LOOKUP, NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP,
    PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_TREE, PARTITION_VERT,
    SS_SIZE_LOOKUP, SUBSIZE_LOOKUP, TX_16X16, TX_32X32, TX_4X4, TX_8X8, TX_MODE_SELECT,
    TX_MODE_TO_BIGGEST_TX_SIZE, TX_SIZE_16_TREE, TX_SIZE_32_TREE, TX_SIZE_8_TREE, ZERO_TOKEN,
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
///
/// キーフレーム・イントラのみを対象とするため、インター予測関連のフィールド
/// （`ref_frame`、動きベクトルなど）は保持しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiInfo {
    pub skip: bool,
    pub tx_size: u8,
    pub mi_size: u8,
    /// `y_mode`（`MiSize >= BLOCK_8X8` の場合は `sub_modes` すべてと同じ値）。
    pub y_mode: u8,
    pub uv_mode: u8,
    /// 8x8 単位内の 4 つの 4x4 サブブロックのイントラモード（`MiSize < BLOCK_8X8` の場合のみ
    /// 複数の異なる値を取り得る）。
    pub sub_modes: [u8; 4],
    pub segment_id: u8,
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
}

impl TileDecoder {
    /// 非圧縮ヘッダと圧縮ヘッダから `TileDecoder` を構築する。
    ///
    /// # Panics
    /// `header.color_config.bit_depth != 8` の場合にパニックする。8bit 以外
    /// （10bit/12bit）のフレームバッファは [`Plane`] が `u8` 固定であるため表現できない。
    /// 呼び出し側（[`crate::decode_keyframe`]）は `TileDecoder::new` を呼ぶ前に
    /// `bit_depth` を検査すること。
    pub fn new(header: &NewFrameHeader, compressed: &CompressedHeader) -> Self {
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
    /// とも矛盾する。本実装ではキーフレームのみを対象とするため、常に固定表
    /// `KF_PARTITION_PROBS` を使用する。
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
        let probs = &KF_PARTITION_PROBS[ctx];

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

        let info = self.intra_frame_mode_info(r, row, col, subsize, avail_u, avail_l)?;
        self.residual(r, row, col, &info, avail_u, avail_l);

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

    /// `residual( )`（仕様 6.4.21 節）。`is_inter` は常に偽（M2 はキーフレームのみ対象）。
    fn residual(
        &mut self,
        r: &mut BoolDecoder,
        row: u32,
        col: u32,
        info: &MiInfo,
        avail_u: bool,
        avail_l: bool,
    ) {
        let bsize = if info.mi_size < BLOCK_8X8 {
            BLOCK_8X8
        } else {
            info.mi_size
        };

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

            let mut block_idx = 0u32;
            let mut y = 0u32;
            while y < num4x4h {
                let mut x = 0u32;
                while x < num4x4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;

                    if start_x < maxx && start_y < maxy {
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

                        if !info.skip {
                            let tx_type = self.compute_tx_type(
                                plane,
                                tx_sz,
                                info.mi_size,
                                info.y_mode,
                                &info.sub_modes,
                                block_idx as usize,
                            );
                            nonzero = self.tokens_and_reconstruct(
                                r,
                                plane,
                                start_x as usize,
                                start_y as usize,
                                tx_sz,
                                tx_type,
                            );
                        }
                    }

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
    }

    /// `get_scan( )`（仕様 6.4.25 節）のうち `TxType` を決定する部分。
    fn compute_tx_type(
        &self,
        plane: usize,
        tx_sz: u8,
        mi_size: u8,
        y_mode: u8,
        sub_modes: &[u8; 4],
        block_idx: usize,
    ) -> TxType {
        if plane > 0 || tx_sz == TX_32X32 {
            TxType::DctDct
        } else if tx_sz == TX_4X4 {
            if self.lossless {
                // is_inter は M2 では常に false なので Lossless のみで判定する。
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
    fn tokens_and_reconstruct(
        &mut self,
        r: &mut BoolDecoder,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: u8,
        tx_type: TxType,
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

            let probs = &self.probs.coef_probs[tx_sz as usize][plane_type][0][band][ctx];

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
        })
    }
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
}
