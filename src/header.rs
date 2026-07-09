//! 非圧縮フレームヘッダ（`uncompressed_header`）のパース（仕様 6.2 節・7.2 節）。
//!
//! 非圧縮フレームヘッダは bool デコーダではなく、[`crate::bit_reader::BitReader`] による
//! 素朴な MSB 優先のビット読み出し（`f(n)` / `s(n)` descriptor、仕様 9.1 節）でパースされる。
//!
//! M1 時点では、以下のみをサポートする:
//! - `show_existing_frame == 1` のケース（既存フレームの再表示指示のみを返す）
//! - `frame_type == KEY_FRAME` のケース（uncompressed_header の全フィールドをパースする）
//!
//! インターフレーム・イントラオンリーフレームのパースは M2 以降で対応する。

use crate::bit_reader::BitReader;

/// `frame_type` syntax element の値（仕様 7.2 節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// frame_type == 0
    KeyFrame,
    /// frame_type == 1
    NonKeyFrame,
}

/// `color_space` の既知の値（仕様 7.2.2 節の表）。
pub const CS_UNKNOWN: u8 = 0;
pub const CS_BT_601: u8 = 1;
pub const CS_BT_709: u8 = 2;
pub const CS_SMPTE_170: u8 = 3;
pub const CS_SMPTE_240: u8 = 4;
pub const CS_BT_2020: u8 = 5;
pub const CS_RESERVED: u8 = 6;
pub const CS_RGB: u8 = 7;

/// ヘッダパース時に発生し得るエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderError {
    /// `frame_marker` が仕様上必須の値 2 ではなかった。
    InvalidFrameMarker,
    /// `frame_sync_code` が仕様上必須の 0x49 0x83 0x42 ではなかった。
    InvalidSyncCode,
    /// `color_space == CS_RGB` かつ `profile_low_bit == 0`
    /// （仕様の適合性要件違反。プロファイル 0 と 2 では RGB は使用できない）。
    InvalidColorConfigForProfile,
    /// `frame_type != KEY_FRAME`。インターフレーム・イントラオンリーフレームは M2 以降で対応する。
    UnsupportedNonKeyFrame,
}

/// ループフィルタ関連パラメータ（仕様 6.2.8 節 `loop_filter_params`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterParams {
    pub level: u8,
    pub sharpness: u8,
    pub delta_enabled: bool,
    /// 参照フレーム種別ごとの調整値。インデックスは
    /// `[INTRA_FRAME, LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME]` の順。
    /// キーフレームでは `setup_past_independence()` により `[1, 0, -1, -1]` が初期値となる
    /// （仕様 7.2 節）。
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

/// 量子化パラメータ（仕様 6.2.9 節 `quantization_params`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i32,
    pub delta_q_uv_dc: i32,
    pub delta_q_uv_ac: i32,
    /// `Lossless = base_q_idx == 0 && delta_q_y_dc == 0 && delta_q_uv_dc == 0 && delta_q_uv_ac == 0`
    pub lossless: bool,
}

/// カラーコンフィグ（仕様 6.2.2 節 `color_config`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    pub bit_depth: u8,
    pub color_space: u8,
    pub color_range: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
}

/// パース済みの非圧縮フレームヘッダ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameHeader {
    /// `show_existing_frame == 1`。新規デコードは行わず、指定インデックスのフレームを表示する。
    ShowExistingFrame { frame_to_show_map_idx: u8 },
    /// 新規にデコードするフレーム（M1 では `frame_type == KEY_FRAME` のみ）。
    New(NewFrameHeader),
}

/// 新規デコードフレームの非圧縮ヘッダフィールド一式。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewFrameHeader {
    pub profile: u8,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub error_resilient_mode: bool,
    pub color_config: ColorConfig,
    pub width: u32,
    pub height: u32,
    pub render_width: u32,
    pub render_height: u32,
    /// 更新対象となる参照フレームスロットのビットマスク。キーフレームでは常に 0xFF。
    pub refresh_frame_flags: u8,
    pub refresh_frame_context: bool,
    pub frame_parallel_decoding_mode: bool,
    pub frame_context_idx: u8,
    pub loop_filter: LoopFilterParams,
    pub quantization: QuantizationParams,
    /// セグメンテーション機能が有効かどうか（詳細パラメータは M2 以降で扱う）。
    pub segmentation_enabled: bool,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    /// 圧縮ヘッダ（`compressed_header`）のバイト数。この直後から
    /// `init_bool(header_size_in_bytes)` で bool デコーダを開始する。
    pub header_size_in_bytes: u16,
}

const MAX_TILE_WIDTH_B64: u32 = 64;
const MIN_TILE_WIDTH_B64: u32 = 4;

/// タイル分割数計算に必要な、フレームサイズから導出される値
/// （仕様 6.2.6 節 `compute_image_size`）。
struct ImageSize {
    sb64_cols: u32,
}

fn compute_image_size(width: u32, height: u32) -> ImageSize {
    let mi_cols = (width + 7) >> 3;
    let mi_rows = (height + 7) >> 3;
    let sb64_cols = (mi_cols + 7) >> 3;
    let _sb64_rows = (mi_rows + 7) >> 3;
    ImageSize { sb64_cols }
}

/// `calc_min_log2_tile_cols()`（仕様 6.2.14 節）。
fn calc_min_log2_tile_cols(sb64_cols: u32) -> u32 {
    let mut min_log2 = 0u32;
    while (MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    min_log2
}

/// `calc_max_log2_tile_cols()`（仕様 6.2.14 節）。
fn calc_max_log2_tile_cols(sb64_cols: u32) -> u32 {
    let mut max_log2 = 1u32;
    while (sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64 {
        max_log2 += 1;
    }
    max_log2 - 1
}

/// `read_prob()`（仕様 6.2.12 節）。
fn read_prob(r: &mut BitReader) -> u8 {
    if r.flag() {
        r.f(8) as u8
    } else {
        255
    }
}

/// `read_delta_q()`（仕様 6.2.10 節）。
fn read_delta_q(r: &mut BitReader) -> i32 {
    if r.flag() {
        r.s(4)
    } else {
        0
    }
}

/// `color_config()`（仕様 6.2.2 節）。
fn parse_color_config(r: &mut BitReader, profile: u8) -> Result<ColorConfig, HeaderError> {
    let bit_depth = if profile >= 2 {
        if r.flag() {
            12
        } else {
            10
        }
    } else {
        8
    };

    let color_space = r.f(3) as u8;

    // 仕様適合性要件: profile_low_bit == 0 (Profile 0 or 2) のとき CS_RGB は使用できない。
    if color_space == CS_RGB && profile & 1 == 0 {
        return Err(HeaderError::InvalidColorConfigForProfile);
    }

    let (color_range, subsampling_x, subsampling_y) = if color_space != CS_RGB {
        let color_range = r.flag();
        let (sx, sy) = if profile == 1 || profile == 3 {
            let sx = r.f(1) as u8;
            let sy = r.f(1) as u8;
            let _reserved_zero = r.f(1);
            (sx, sy)
        } else {
            (1u8, 1u8)
        };
        (color_range, sx, sy)
    } else {
        if profile == 1 || profile == 3 {
            let _reserved_zero = r.f(1);
        }
        (true, 0u8, 0u8)
    };

    Ok(ColorConfig {
        bit_depth,
        color_space,
        color_range,
        subsampling_x,
        subsampling_y,
    })
}

/// `frame_size()` + `compute_image_size()`（仕様 6.2.3 節・6.2.6 節）。
fn parse_frame_size(r: &mut BitReader) -> (u32, u32) {
    let frame_width_minus_1 = r.f(16);
    let frame_height_minus_1 = r.f(16);
    (frame_width_minus_1 + 1, frame_height_minus_1 + 1)
}

/// `render_size()`（仕様 6.2.4 節）。
fn parse_render_size(r: &mut BitReader, width: u32, height: u32) -> (u32, u32) {
    if r.flag() {
        let render_width_minus_1 = r.f(16);
        let render_height_minus_1 = r.f(16);
        (render_width_minus_1 + 1, render_height_minus_1 + 1)
    } else {
        (width, height)
    }
}

/// `loop_filter_params()`（仕様 6.2.8 節）。
fn parse_loop_filter_params(r: &mut BitReader) -> LoopFilterParams {
    let level = r.f(6) as u8;
    let sharpness = r.f(3) as u8;
    let delta_enabled = r.flag();

    // setup_past_independence() によるキーフレームでの初期値（仕様 7.2 節）。
    let mut ref_deltas: [i8; 4] = [1, 0, -1, -1];
    let mut mode_deltas: [i8; 2] = [0, 0];

    if delta_enabled {
        let delta_update = r.flag();
        if delta_update {
            for delta in ref_deltas.iter_mut() {
                if r.flag() {
                    *delta = r.s(6) as i8;
                }
            }
            for delta in mode_deltas.iter_mut() {
                if r.flag() {
                    *delta = r.s(6) as i8;
                }
            }
        }
    }

    LoopFilterParams {
        level,
        sharpness,
        delta_enabled,
        ref_deltas,
        mode_deltas,
    }
}

/// `quantization_params()`（仕様 6.2.9 節）。
fn parse_quantization_params(r: &mut BitReader) -> QuantizationParams {
    let base_q_idx = r.f(8) as u8;
    let delta_q_y_dc = read_delta_q(r);
    let delta_q_uv_dc = read_delta_q(r);
    let delta_q_uv_ac = read_delta_q(r);
    let lossless = base_q_idx == 0 && delta_q_y_dc == 0 && delta_q_uv_dc == 0 && delta_q_uv_ac == 0;

    QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        delta_q_uv_dc,
        delta_q_uv_ac,
        lossless,
    }
}

const SEG_LVL_MAX: usize = 4;
const MAX_SEGMENTS: usize = 8;
const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 2, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];

/// `segmentation_params()`（仕様 6.2.11 節）。
///
/// M1 では `segmentation_enabled` の有無のみを [`NewFrameHeader`] に保持するが、
/// 後続の `tile_info()` / `header_size_in_bytes` を正しい位置から読むために、
/// セグメンテーションのペイロード全体をビット単位で正しく読み進める。
fn parse_segmentation_params(r: &mut BitReader) -> bool {
    let segmentation_enabled = r.flag();
    if !segmentation_enabled {
        return false;
    }

    let segmentation_update_map = r.flag();
    if segmentation_update_map {
        for _ in 0..7 {
            let _segmentation_tree_prob = read_prob(r);
        }
        let segmentation_temporal_update = r.flag();
        for _ in 0..3 {
            if segmentation_temporal_update {
                let _segmentation_pred_prob = read_prob(r);
            }
            // temporal_update == 0 の場合、prob = 255 でビットは消費しない。
        }
    }

    let segmentation_update_data = r.flag();
    if segmentation_update_data {
        let _segmentation_abs_or_delta_update = r.flag();
        for _segment in 0..MAX_SEGMENTS {
            for level in 0..SEG_LVL_MAX {
                let feature_enabled = r.flag();
                if feature_enabled {
                    let bits_to_read = SEGMENTATION_FEATURE_BITS[level];
                    if bits_to_read > 0 {
                        let _feature_value = r.f(bits_to_read);
                    }
                    if SEGMENTATION_FEATURE_SIGNED[level] {
                        let _feature_sign = r.f(1);
                    }
                }
            }
        }
    }

    true
}

/// `tile_info()`（仕様 6.2.13 節）。戻り値は `(tile_cols_log2, tile_rows_log2)`。
fn parse_tile_info(r: &mut BitReader, sb64_cols: u32) -> (u32, u32) {
    let min_log2_tile_cols = calc_min_log2_tile_cols(sb64_cols);
    let max_log2_tile_cols = calc_max_log2_tile_cols(sb64_cols);

    let mut tile_cols_log2 = min_log2_tile_cols;
    while tile_cols_log2 < max_log2_tile_cols {
        if r.flag() {
            tile_cols_log2 += 1;
        } else {
            break;
        }
    }

    let mut tile_rows_log2 = r.f(1);
    if tile_rows_log2 == 1 {
        let increment_tile_rows_log2 = r.f(1);
        tile_rows_log2 += increment_tile_rows_log2;
    }

    (tile_cols_log2, tile_rows_log2)
}

/// `uncompressed_header()`（仕様 6.2 節）をパースする。
///
/// 戻り値はパース結果と、`trailing_bits()` によるバイト境界揃えまで含めた消費バイト数の組。
/// `show_existing_frame == 1` の場合、または `frame_type == KEY_FRAME` の場合にのみ成功する。
pub fn parse_uncompressed_header(data: &[u8]) -> Result<(FrameHeader, usize), HeaderError> {
    let mut r = BitReader::new(data);

    let frame_marker = r.f(2);
    if frame_marker != 2 {
        return Err(HeaderError::InvalidFrameMarker);
    }

    let profile_low_bit = r.f(1);
    let profile_high_bit = r.f(1);
    let profile = ((profile_high_bit << 1) + profile_low_bit) as u8;
    if profile == 3 {
        let _reserved_zero = r.f(1);
    }

    let show_existing_frame = r.flag();
    if show_existing_frame {
        let frame_to_show_map_idx = r.f(3) as u8;
        let consumed = r.byte_position_ceil();
        return Ok((
            FrameHeader::ShowExistingFrame {
                frame_to_show_map_idx,
            },
            consumed,
        ));
    }

    let frame_type = if r.f(1) == 0 {
        FrameType::KeyFrame
    } else {
        FrameType::NonKeyFrame
    };
    let show_frame = r.flag();
    let error_resilient_mode = r.flag();

    if frame_type != FrameType::KeyFrame {
        // インターフレーム・イントラオンリーフレームは M2 以降で対応する。
        return Err(HeaderError::UnsupportedNonKeyFrame);
    }

    // frame_sync_code()
    let sync = [r.f(8), r.f(8), r.f(8)];
    if sync != [0x49, 0x83, 0x42] {
        return Err(HeaderError::InvalidSyncCode);
    }

    let color_config = parse_color_config(&mut r, profile)?;
    let (width, height) = parse_frame_size(&mut r);
    let (render_width, render_height) = parse_render_size(&mut r, width, height);
    // refresh_frame_flags はビットストリームからは読まず、キーフレームでは常に 0xFF が代入される。
    let refresh_frame_flags = 0xFFu8;

    let (refresh_frame_context, frame_parallel_decoding_mode) = if !error_resilient_mode {
        (r.flag(), r.flag())
    } else {
        (false, true)
    };
    let _frame_context_idx_raw = r.f(2) as u8;
    // FrameIsIntra == 1 (キーフレーム) のため setup_past_independence() が必ず呼ばれ、
    // 仕様上 frame_context_idx はここで 0 にリセットされる。以降の非圧縮ヘッダの
    // パースにはこの値は影響しないため、フィールドにはビットストリームの生の値ではなく
    // リセット後の値 (0) を保持する。
    let frame_context_idx = 0u8;

    let loop_filter = parse_loop_filter_params(&mut r);
    let quantization = parse_quantization_params(&mut r);
    let segmentation_enabled = parse_segmentation_params(&mut r);

    let image_size = compute_image_size(width, height);
    let (tile_cols_log2, tile_rows_log2) = parse_tile_info(&mut r, image_size.sb64_cols);

    let header_size_in_bytes = r.f(16) as u16;

    let consumed = r.byte_position_ceil();

    Ok((
        FrameHeader::New(NewFrameHeader {
            profile,
            frame_type,
            show_frame,
            error_resilient_mode,
            color_config,
            width,
            height,
            render_width,
            render_height,
            refresh_frame_flags,
            refresh_frame_context,
            frame_parallel_decoding_mode,
            frame_context_idx,
            loop_filter,
            quantization,
            segmentation_enabled,
            tile_cols_log2,
            tile_rows_log2,
            header_size_in_bytes,
        }),
        consumed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の MSB 優先ビットライター。手組みのビットストリームを組み立てるために使う。
    struct BitWriter {
        bytes: Vec<u8>,
        cur: u8,
        cur_bits: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                cur: 0,
                cur_bits: 0,
            }
        }

        fn push_bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                let bit = ((value >> i) & 1) as u8;
                self.cur = (self.cur << 1) | bit;
                self.cur_bits += 1;
                if self.cur_bits == 8 {
                    self.bytes.push(self.cur);
                    self.cur = 0;
                    self.cur_bits = 0;
                }
            }
        }

        fn push_flag(&mut self, value: bool) {
            self.push_bits(value as u32, 1);
        }

        /// s(n): 絶対値 n ビット + 符号 1 ビット。
        fn push_signed(&mut self, value: i32, n: u32) {
            self.push_bits(value.unsigned_abs(), n);
            self.push_flag(value < 0);
        }

        fn finish(mut self) -> Vec<u8> {
            while self.cur_bits != 0 {
                self.cur <<= 1;
                self.cur_bits += 1;
                if self.cur_bits == 8 {
                    self.bytes.push(self.cur);
                    self.cur = 0;
                    self.cur_bits = 0;
                }
            }
            self.bytes
        }
    }

    /// 最小構成のキーフレーム非圧縮ヘッダを組み立てる。
    /// profile=0, 8x8, ロスレス, セグメンテーション無効, タイル分割なし。
    fn build_minimal_keyframe_header() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_bits(2, 2); // frame_marker
        w.push_bits(0, 1); // profile_low_bit
        w.push_bits(0, 1); // profile_high_bit -> profile=0
        w.push_flag(false); // show_existing_frame
        w.push_bits(0, 1); // frame_type = KEY_FRAME
        w.push_flag(true); // show_frame
        w.push_flag(false); // error_resilient_mode
        w.push_bits(0x49, 8);
        w.push_bits(0x83, 8);
        w.push_bits(0x42, 8);
        // color_config (profile 0 -> bit_depth=8 は読まない)
        w.push_bits(CS_UNKNOWN as u32, 3); // color_space
        w.push_flag(false); // color_range
                            // profile 0 -> subsampling は読まない
                            // frame_size
        w.push_bits(7, 16); // frame_width_minus_1 -> width=8
        w.push_bits(7, 16); // frame_height_minus_1 -> height=8
                            // render_size
        w.push_flag(false); // render_and_frame_size_different = 0
                            // error_resilient_mode == 0 -> refresh_frame_context, frame_parallel_decoding_mode
        w.push_flag(true); // refresh_frame_context
        w.push_flag(false); // frame_parallel_decoding_mode
        w.push_bits(0, 2); // frame_context_idx
                           // loop_filter_params
        w.push_bits(0, 6); // loop_filter_level
        w.push_bits(0, 3); // loop_filter_sharpness
        w.push_flag(false); // loop_filter_delta_enabled
                            // quantization_params (すべて 0 -> lossless)
        w.push_bits(0, 8); // base_q_idx
        w.push_flag(false); // delta_q_y_dc coded?
        w.push_flag(false); // delta_q_uv_dc coded?
        w.push_flag(false); // delta_q_uv_ac coded?
                            // segmentation_params
        w.push_flag(false); // segmentation_enabled
                            // tile_info: width=8 -> MiCols=1, Sb64Cols=1 -> min_log2=0, max_log2=0 -> ループ回らず
        w.push_bits(0, 1); // tile_rows_log2 = 0
                           // header_size_in_bytes
        w.push_bits(1, 16);

        w.finish()
    }

    #[test]
    fn parses_minimal_keyframe_header() {
        let data = build_minimal_keyframe_header();
        let (header, _consumed) = parse_uncompressed_header(&data).expect("should parse");
        match header {
            FrameHeader::New(f) => {
                assert_eq!(f.profile, 0);
                assert_eq!(f.frame_type, FrameType::KeyFrame);
                assert!(f.show_frame);
                assert!(!f.error_resilient_mode);
                assert_eq!(f.color_config.bit_depth, 8);
                assert_eq!(f.color_config.color_space, CS_UNKNOWN);
                assert_eq!(f.color_config.subsampling_x, 1);
                assert_eq!(f.color_config.subsampling_y, 1);
                assert_eq!(f.width, 8);
                assert_eq!(f.height, 8);
                assert_eq!(f.render_width, 8);
                assert_eq!(f.render_height, 8);
                assert_eq!(f.refresh_frame_flags, 0xFF);
                assert!(f.quantization.lossless);
                assert!(!f.segmentation_enabled);
                assert_eq!(f.tile_cols_log2, 0);
                assert_eq!(f.tile_rows_log2, 0);
                assert_eq!(f.header_size_in_bytes, 1);
                assert_eq!(f.loop_filter.ref_deltas, [1, 0, -1, -1]);
            }
            FrameHeader::ShowExistingFrame { .. } => panic!("unexpected show_existing_frame"),
        }
    }

    #[test]
    fn rejects_bad_frame_marker() {
        let mut w = BitWriter::new();
        w.push_bits(1, 2); // frame_marker != 2
        w.push_bits(0, 30);
        let data = w.finish();
        assert_eq!(
            parse_uncompressed_header(&data),
            Err(HeaderError::InvalidFrameMarker)
        );
    }

    #[test]
    fn rejects_bad_sync_code() {
        let mut w = BitWriter::new();
        w.push_bits(2, 2);
        w.push_bits(0, 1);
        w.push_bits(0, 1);
        w.push_flag(false); // show_existing_frame
        w.push_bits(0, 1); // KEY_FRAME
        w.push_flag(true);
        w.push_flag(false);
        w.push_bits(0x00, 8); // 不正な sync byte
        w.push_bits(0x00, 8);
        w.push_bits(0x00, 8);
        let data = w.finish();
        assert_eq!(
            parse_uncompressed_header(&data),
            Err(HeaderError::InvalidSyncCode)
        );
    }

    #[test]
    fn rejects_non_key_frame() {
        let mut w = BitWriter::new();
        w.push_bits(2, 2);
        w.push_bits(0, 1);
        w.push_bits(0, 1);
        w.push_flag(false); // show_existing_frame
        w.push_bits(1, 1); // NON_KEY_FRAME
        w.push_flag(true);
        w.push_flag(false);
        let data = w.finish();
        assert_eq!(
            parse_uncompressed_header(&data),
            Err(HeaderError::UnsupportedNonKeyFrame)
        );
    }

    #[test]
    fn parses_show_existing_frame() {
        let mut w = BitWriter::new();
        w.push_bits(2, 2); // frame_marker
        w.push_bits(0, 1); // profile_low_bit
        w.push_bits(0, 1); // profile_high_bit
        w.push_flag(true); // show_existing_frame = 1
        w.push_bits(5, 3); // frame_to_show_map_idx = 5
        let data = w.finish();

        let (header, consumed) = parse_uncompressed_header(&data).expect("should parse");
        assert_eq!(
            header,
            FrameHeader::ShowExistingFrame {
                frame_to_show_map_idx: 5
            }
        );
        assert_eq!(consumed, 1);
    }

    #[test]
    fn parses_loop_filter_deltas_and_signed_values() {
        let mut w = BitWriter::new();
        w.push_bits(2, 2);
        w.push_bits(0, 1);
        w.push_bits(0, 1);
        w.push_flag(false); // show_existing_frame
        w.push_bits(0, 1); // KEY_FRAME
        w.push_flag(true);
        w.push_flag(true); // error_resilient_mode = 1
        w.push_bits(0x49, 8);
        w.push_bits(0x83, 8);
        w.push_bits(0x42, 8);
        w.push_bits(CS_UNKNOWN as u32, 3);
        w.push_flag(false);
        w.push_bits(15, 16); // width = 16
        w.push_bits(15, 16); // height = 16
        w.push_flag(false); // render_size same as frame
                            // error_resilient_mode == 1 -> refresh_frame_context/frame_parallel_decoding_mode は読まない
        w.push_bits(0, 2); // frame_context_idx
                           // loop_filter_params
        w.push_bits(10, 6); // level
        w.push_bits(3, 3); // sharpness
        w.push_flag(true); // delta_enabled
        w.push_flag(true); // delta_update
                           // update_ref_delta x4
        w.push_flag(true);
        w.push_signed(-3, 6);
        w.push_flag(false);
        w.push_flag(true);
        w.push_signed(2, 6);
        w.push_flag(false);
        // update_mode_delta x2
        w.push_flag(true);
        w.push_signed(-1, 6);
        w.push_flag(false);
        // quantization_params
        w.push_bits(20, 8); // base_q_idx (lossless ではない)
        w.push_flag(false);
        w.push_flag(false);
        w.push_flag(false);
        // segmentation
        w.push_flag(false);
        // tile_info: width=16 -> MiCols=2, Sb64Cols=1 -> 同上でループなし
        w.push_bits(0, 1);
        w.push_bits(42, 16); // header_size_in_bytes

        let data = w.finish();
        let (header, _) = parse_uncompressed_header(&data).expect("should parse");
        match header {
            FrameHeader::New(f) => {
                assert!(f.error_resilient_mode);
                assert!(!f.refresh_frame_context);
                assert!(f.frame_parallel_decoding_mode);
                assert_eq!(f.loop_filter.level, 10);
                assert_eq!(f.loop_filter.sharpness, 3);
                assert_eq!(f.loop_filter.ref_deltas, [-3, 0, 2, -1]);
                assert_eq!(f.loop_filter.mode_deltas, [-1, 0]);
                assert!(!f.quantization.lossless);
                assert_eq!(f.header_size_in_bytes, 42);
            }
            FrameHeader::ShowExistingFrame { .. } => panic!("unexpected"),
        }
    }
}
