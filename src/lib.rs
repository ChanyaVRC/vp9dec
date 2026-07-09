//! vp9dec: 完全自作の VP9 動画デコーダ（依存クレートゼロ）。
//!
//! 参照仕様: VP9 Bitstream & Decoding Process Specification v0.7
//! (Google, 2017年2月22日版, <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>)
//!
//! # マイルストーン
//! - M1: IVF コンテナパーサ、bool デコーダ、非圧縮フレームヘッダのパース
//! - M2: イントラ予測によるキーフレームのデコード
//! - M3: インター予測（動き補償）
//! - M4: コンフォーマンステスト完全通過
//!
//! （モジュールは以降のコミットで段階的に追加する。）

pub mod bit_reader;
pub mod bool_coder;
pub mod compressed_header;
pub mod framebuffer;
pub mod header;
pub mod ivf;
pub mod predict;
pub mod prob_tables;
pub mod quant;
pub mod scan;
pub mod tile;
pub mod transform;

use compressed_header::{parse_compressed_header, CompressedHeaderError};
use header::{parse_uncompressed_header, FrameHeader, HeaderError};
use tile::{TileDecoder, TileError};

/// [`decode_keyframe`] が失敗し得るすべての要因をまとめたエラー型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// 非圧縮ヘッダ（`uncompressed_header`）のパースに失敗した。
    Header(HeaderError),
    /// 圧縮ヘッダ（`compressed_header`）のパースに失敗した。
    CompressedHeader(CompressedHeaderError),
    /// タイル・モード情報・トークン復号に失敗した。
    Tile(TileError),
    /// `show_existing_frame == 1` のフレームは `decode_keyframe` の対象外
    /// （新規デコードを行わず、既存フレームの再表示指示のみを含むため）。
    ShowExistingFrameNotSupported,
    /// `header_size_in_bytes` がフレームデータ長を超えるなど、フレームデータが不正。
    TruncatedFrame,
    /// 8bit（`BitDepth == 8`）以外のフレーム。[`framebuffer::Plane`] が `u8` 固定のため、
    /// 10bit/12bit フレームは現時点ではサポート対象外。
    UnsupportedBitDepth(u8),
}

impl From<HeaderError> for DecodeError {
    fn from(e: HeaderError) -> Self {
        DecodeError::Header(e)
    }
}

impl From<CompressedHeaderError> for DecodeError {
    fn from(e: CompressedHeaderError) -> Self {
        DecodeError::CompressedHeader(e)
    }
}

impl From<TileError> for DecodeError {
    fn from(e: TileError) -> Self {
        DecodeError::Tile(e)
    }
}

/// デコード結果の 1 フレーム。表示サイズ（`FrameWidth`/`FrameHeight`）にクロップ済みで、
/// YUV420 の 3 プレーンを行優先（row-major）の `Vec<u8>` として保持する。
///
/// `u`/`v` のサイズは仕様 8.9 節の出力プロセスに従い、
/// `((width + subsampling_x) >> subsampling_x) x ((height + subsampling_y) >> subsampling_y)`
/// で計算される（4:2:0 の場合は `((width+1)/2) x ((height+1)/2)`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 1 枚のキーフレームをデコードする。
///
/// `frame_data` は IVF 等のコンテナから取り出した、1 フレーム分の VP9 ビットストリーム
/// （`uncompressed_header` から始まる生データ）。`show_existing_frame == 1` の場合や
/// `frame_type != KEY_FRAME` の場合はエラーを返す（M2 はキーフレームのイントラ復号のみを
/// 対象とするため）。
///
/// ループフィルタ（仕様 8.8 節）は M2b で実装予定のため未適用。ブロックノイズが
/// 残る場合がある（詳細は README.md を参照）。
pub fn decode_keyframe(frame_data: &[u8]) -> Result<Frame, DecodeError> {
    let (parsed, consumed) = parse_uncompressed_header(frame_data)?;
    let header = match parsed {
        FrameHeader::New(h) => h,
        FrameHeader::ShowExistingFrame { .. } => {
            return Err(DecodeError::ShowExistingFrameNotSupported)
        }
    };

    if header.color_config.bit_depth != 8 {
        return Err(DecodeError::UnsupportedBitDepth(
            header.color_config.bit_depth,
        ));
    }

    let header_size = header.header_size_in_bytes as usize;
    let compressed_start = consumed;
    let compressed_end = compressed_start
        .checked_add(header_size)
        .ok_or(DecodeError::TruncatedFrame)?;
    if compressed_end > frame_data.len() {
        return Err(DecodeError::TruncatedFrame);
    }
    let compressed_bytes = &frame_data[compressed_start..compressed_end];
    let compressed = parse_compressed_header(compressed_bytes, header.quantization.lossless)?;

    let tile_data = &frame_data[compressed_end..];
    let mut decoder = TileDecoder::new(&header, &compressed);
    decoder.decode_tiles(tile_data)?;

    let planes = decoder.planes();
    let width = header.width as usize;
    let height = header.height as usize;
    let sub_x = header.color_config.subsampling_x as u32;
    let sub_y = header.color_config.subsampling_y as u32;
    let uv_width = ((width as u32 + sub_x) >> sub_x) as usize;
    let uv_height = ((height as u32 + sub_y) >> sub_y) as usize;

    Ok(Frame {
        width: header.width,
        height: header.height,
        y: planes[0].crop(width, height),
        u: planes[1].crop(uv_width, uv_height),
        v: planes[2].crop(uv_width, uv_height),
    })
}
