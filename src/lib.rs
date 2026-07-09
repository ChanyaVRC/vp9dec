//! vp9dec: 完全自作の VP9 動画デコーダ（依存クレートゼロ）。
//!
//! 参照仕様: VP9 Bitstream & Decoding Process Specification v0.7
//! (Google, 2017年2月22日版, <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>)
//!
//! # マイルストーン
//! - M1: IVF コンテナパーサ、bool デコーダ、非圧縮フレームヘッダのパース
//! - M2: イントラ予測によるキーフレームのデコード
//! - M2b: ループフィルタ（デブロッキングフィルタ）＋公式コンフォーマンス検証
//! - M3 前半: インターフレームのビットストリーム復号（動き補償の手前まで）
//! - M3 後半: 動き補償・確率適応・参照フレーム管理＋全フレーム MD5 コンフォーマンス
//! - M4: コンフォーマンステスト完全通過
//!
//! （モジュールは以降のコミットで段階的に追加する。）

pub mod bit_reader;
pub mod bool_coder;
pub mod compressed_header;
pub mod counts;
pub mod dpb;
pub mod framebuffer;
pub mod header;
pub mod ivf;
pub mod loop_filter;
pub mod md5;
pub mod mv;
pub mod predict;
pub mod prob_tables;
pub mod quant;
pub mod scan;
pub mod tile;
pub mod transform;

use compressed_header::{parse_compressed_header_ex, CompressedHeaderError, FrameContextStore};
use counts::{adapt_coef_probs, adapt_noncoef_probs};
use dpb::{Dpb, RefFrameData};
use header::{
    parse_uncompressed_header, ColorConfig, FrameHeader, FrameType, HeaderError, NUM_REF_FRAMES,
};
use tile::{MiGrid, TileDecoder, TileError};

/// [`decode_keyframe`]/[`Decoder::decode_frame`] が失敗し得るすべての要因をまとめたエラー型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// 非圧縮ヘッダ（`uncompressed_header`）のパースに失敗した。
    Header(HeaderError),
    /// 圧縮ヘッダ（`compressed_header`）のパースに失敗した。
    CompressedHeader(CompressedHeaderError),
    /// タイル・モード情報・トークン復号に失敗した。
    Tile(TileError),
    /// [`decode_keyframe`] にキーフレーム以外（インターフレーム・イントラオンリーフレーム）
    /// が渡された。
    NotAKeyFrame,
    /// `header_size_in_bytes` がフレームデータ長を超えるなど、フレームデータが不正。
    TruncatedFrame,
    /// 8bit（`BitDepth == 8`）以外のフレーム。[`framebuffer::Plane`] が `u8` 固定のため、
    /// 10bit/12bit フレームは現時点ではサポート対象外。
    UnsupportedBitDepth(u8),
    /// `show_existing_frame` が指す DPB スロットにフレームが格納されていない
    /// （通常のコンフォーマンスビットストリームでは発生しない）。
    MissingReferenceFrame,
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

/// フレームバッファ（`planes`）を表示サイズにクロップして [`Frame`] を組み立てる。
fn crop_to_frame(
    planes: &[framebuffer::Plane; 3],
    width: u32,
    height: u32,
    color_config: &ColorConfig,
) -> Frame {
    let sub_x = color_config.subsampling_x as u32;
    let sub_y = color_config.subsampling_y as u32;
    let uv_width = ((width + sub_x) >> sub_x) as usize;
    let uv_height = ((height + sub_y) >> sub_y) as usize;

    Frame {
        width,
        height,
        y: planes[0].crop(width as usize, height as usize),
        u: planes[1].crop(uv_width, uv_height),
        v: planes[2].crop(uv_width, uv_height),
    }
}

/// [`crop_to_frame`] と同じクロップ計算で、DPB 格納用の [`RefFrameData`] を組み立てる
/// （仕様 8.10 節 "Reference frame update process" ステップ 1）。
fn build_ref_frame_data(
    planes: &[framebuffer::Plane; 3],
    width: u32,
    height: u32,
    color_config: &ColorConfig,
) -> RefFrameData {
    let sub_x = color_config.subsampling_x as u32;
    let sub_y = color_config.subsampling_y as u32;
    let uv_width = ((width + sub_x) >> sub_x) as usize;
    let uv_height = ((height + sub_y) >> sub_y) as usize;

    RefFrameData {
        width,
        height,
        subsampling_x: sub_x,
        subsampling_y: sub_y,
        bit_depth: color_config.bit_depth,
        y: planes[0].crop_to_plane(width as usize, height as usize),
        u: planes[1].crop_to_plane(uv_width, uv_height),
        v: planes[2].crop_to_plane(uv_width, uv_height),
    }
}

fn ref_frame_data_to_frame(data: &RefFrameData) -> Frame {
    Frame {
        width: data.width,
        height: data.height,
        y: data.y.crop(data.y.width, data.y.height),
        u: data.u.crop(data.u.width, data.u.height),
        v: data.v.crop(data.v.width, data.v.height),
    }
}

/// 1 枚のキーフレームをデコードする。
///
/// `frame_data` は IVF 等のコンテナから取り出した、1 フレーム分の VP9 ビットストリーム
/// （`uncompressed_header` から始まる生データ）。`show_existing_frame == 1` の場合や
/// `frame_type != KEY_FRAME` の場合はエラーを返す。
///
/// キーフレームは他フレームを参照しない・確率テーブルを常にデフォルトから開始するため、
/// フレーム間状態（[`Decoder`]）を持たない単発の関数として提供している
/// （内部的には使い捨ての [`Decoder`] を介して [`Decoder::decode_frame`] を呼ぶ）。
/// 複数フレーム（インターフレームを含む）を順に読み進める場合は [`Decoder`] を直接使うこと。
///
/// タイル復号後、クロップ前にループフィルタ（仕様 8.8 節、[`crate::loop_filter`]）を適用する。
pub fn decode_keyframe(frame_data: &[u8]) -> Result<Frame, DecodeError> {
    // キーフレームであることを事前に確認する（decode_frame は非キーフレームも受け付けるため）。
    let dummy_ref_sizes = [(0u32, 0u32); NUM_REF_FRAMES];
    let dummy_lf_deltas = (
        header::DEFAULT_LOOP_FILTER_REF_DELTAS,
        header::DEFAULT_LOOP_FILTER_MODE_DELTAS,
    );
    let (parsed, _consumed) =
        parse_uncompressed_header(frame_data, &dummy_ref_sizes, dummy_lf_deltas)?;
    match &parsed {
        FrameHeader::New(h) if h.frame_type == FrameType::KeyFrame => {}
        FrameHeader::New(_) => return Err(DecodeError::NotAKeyFrame),
        FrameHeader::ShowExistingFrame { .. } => return Err(DecodeError::NotAKeyFrame),
    }

    let mut decoder = Decoder::new();
    match decoder.decode_frame(frame_data)? {
        Some(frame) => Ok(frame),
        None => unreachable!("キーフレームは常に show_frame == 1（frame_is_intra の要件）"),
    }
}

/// 複数フレームを順にデコードするための状態付きデコーダ。
///
/// VP9 はフレーム間で以下の状態を引き継ぐため、1 フレームずつ独立に処理できない
/// （`decode_keyframe` はキーフレーム単体のみを扱うため、この状態を持たない）:
/// - 参照フレームスロット（`RefFrameWidth`/`RefFrameHeight`、仕様 6.2.5 節
///   `frame_size_with_refs`）と実ピクセルデータ（[`Dpb`]、仕様 8.10 節）。
/// - フレームコンテキスト（`frame_context_idx` で選択される確率テーブル 4 スロット、
///   仕様 7.1.2 節 `load_probs`/`save_probs`）。[`FrameContextStore`] 参照。
/// - `UsePrevFrameMvs`（仕様 7.2.6 節）が真のとき参照する前フレームの `Mvs`/`RefFrames`
///   （本実装では前フレームの [`MiGrid`] をまるごと保持する）。
/// - ループフィルタの `ref_deltas`/`mode_deltas`（仕様 7.2 節、`setup_past_independence()`
///   でのみリセットされる）。
/// - `LastFrameType`（仕様 7.2 節。確率適応の `updateFactor` 計算に使う、仕様 8.4.3 節）。
pub struct Decoder {
    ref_frame_sizes: [(u32, u32); NUM_REF_FRAMES],
    frame_contexts: FrameContextStore,
    /// `UsePrevFrameMvs` 計算用（仕様 7.2.6 節）。`show_existing_frame` では更新されない
    /// （`compute_image_size` が呼ばれないため）。
    prev_frame_dims: Option<(u32, u32)>,
    prev_show_frame: Option<bool>,
    /// 前フレームの `Mvs`/`RefFrames`（`PrevMvs`/`PrevRefFrames` 相当）。
    prev_mi_grid: Option<MiGrid>,
    /// インターフレーム・イントラオンリーフレーム（`Profile == 0`）は `color_config` を
    /// ビットストリームで再送しないため、直近のキーフレーム/イントラオンリーフレームの
    /// 値を引き継ぐ（仕様 7.2 節: 参照フレームとビット深度・サブサンプリングが一致することが
    /// 適合性要件）。
    last_color_config: Option<ColorConfig>,
    /// 参照フレームの実ピクセルデータ（仕様 8.10 節 `FrameStore`）。
    dpb: Dpb,
    /// ループフィルタの `ref_deltas`/`mode_deltas`（仕様 7.2 節で持続する状態）。
    loop_filter_deltas: ([i8; 4], [i8; 2]),
    /// `LastFrameType`（仕様 7.2 節）。`show_existing_frame` フレームでは更新されない。
    last_frame_type: Option<FrameType>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            ref_frame_sizes: [(0, 0); NUM_REF_FRAMES],
            frame_contexts: FrameContextStore::new(),
            prev_frame_dims: None,
            prev_show_frame: None,
            prev_mi_grid: None,
            last_color_config: None,
            dpb: Dpb::new(),
            loop_filter_deltas: (
                header::DEFAULT_LOOP_FILTER_REF_DELTAS,
                header::DEFAULT_LOOP_FILTER_MODE_DELTAS,
            ),
            last_frame_type: None,
        }
    }

    /// 1 フレーム分の VP9 ビットストリームをデコードする。呼び出し側は IVF などのコンテナ
    /// から取り出したフレームを表示順ではなくビットストリーム順（デコード順）に渡すこと。
    ///
    /// 戻り値は「表示すべきフレームが得られたか」を表す: `show_existing_frame == 1` または
    /// `show_frame == 1` の場合は `Some(Frame)`、それ以外（`show_frame == 0` の非表示
    /// フレーム、いわゆる droppable/altref フレーム）の場合は `None` を返す。内部状態
    /// （参照フレームバッファ・フレームコンテキスト・前フレームの MV 等）はどちらの場合も
    /// 正しく更新される。
    pub fn decode_frame(&mut self, frame_data: &[u8]) -> Result<Option<Frame>, DecodeError> {
        let (parsed, consumed) =
            parse_uncompressed_header(frame_data, &self.ref_frame_sizes, self.loop_filter_deltas)?;
        let mut header = match parsed {
            FrameHeader::New(h) => h,
            FrameHeader::ShowExistingFrame {
                frame_to_show_map_idx,
            } => {
                let data = self
                    .dpb
                    .get(frame_to_show_map_idx)
                    .ok_or(DecodeError::MissingReferenceFrame)?;
                return Ok(Some(ref_frame_data_to_frame(data)));
            }
        };

        if header.frame_is_intra {
            self.last_color_config = Some(header.color_config);
        } else if let Some(cc) = self.last_color_config {
            // インターフレームは color_config を再送しない（header.rs のコメント参照）。
            header.color_config = cc;
        }

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

        // UsePrevFrameMvs（仕様 7.2.6 節）。
        let use_prev_frame_mvs = !header.frame_is_intra
            && !header.error_resilient_mode
            && self.prev_frame_dims == Some((header.width, header.height))
            && self.prev_show_frame == Some(true);

        // setup_past_independence()（仕様 7.2 節）: FrameIsIntra || error_resilient_mode の
        // 場合、全確率テーブルをデフォルトへリセットしたうえで frame_context_idx は 0 に固定
        // される（header.rs 側で既に 0 に補正済み）。reset_frame_context による部分リセット
        // （2: 該当スロットのみ）は未実装だが、後続の load はどのみち header.frame_context_idx
        // （常に 0）から行われるため、ビットストリーム読み取りの結果には影響しない。
        if header.frame_is_intra || header.error_resilient_mode {
            self.frame_contexts.reset_all();
        }
        // `load_probs`/`load_probs2`（仕様 6.1 節 `frame()` 冒頭）に相当する開始時点の値。
        // `refresh_probs()`（仕様 6.1.2 節）で forward update 前のこの値へ戻したうえで
        // backward adaptation を適用するため、compressed_header 呼び出し後も保持しておく。
        let starting_probs = self.frame_contexts.load(header.frame_context_idx);

        let compressed = parse_compressed_header_ex(
            compressed_bytes,
            header.quantization.lossless,
            header.frame_is_intra,
            header.interpolation_filter,
            header.ref_frame_sign_bias,
            header.allow_high_precision_mv,
            starting_probs.clone(),
        )?;

        // 動き補償用に、このフレームが参照する DPB スロットのピクセルデータを解決する
        // （仕様 8.5.2.3〜8.5.2.4 節）。`FrameIsIntra == 1` の場合は参照しない。
        let resolved_refs: [Option<RefFrameData>; 3] = if header.frame_is_intra {
            [None, None, None]
        } else {
            std::array::from_fn(|i| self.dpb.get(header.ref_frame_idx[i]).cloned())
        };

        let tile_data = &frame_data[compressed_end..];
        let prev_grid = if use_prev_frame_mvs {
            self.prev_mi_grid.clone()
        } else {
            None
        };
        let mut tile_decoder = TileDecoder::new_with_prev(
            &header,
            &compressed,
            use_prev_frame_mvs,
            prev_grid,
            resolved_refs,
        );
        tile_decoder.decode_tiles(tile_data)?;
        tile_decoder.apply_loop_filter(&header.loop_filter);

        // refresh_probs()（仕様 6.1.2 節）。
        let final_probs = if !header.error_resilient_mode && !header.frame_parallel_decoding_mode {
            // load_probs( frame_context_idx ): tx_probs/skip_prob を除くすべてのテーブルを
            // forward update 前の値（starting_probs）へ戻す。tx_probs/skip_prob は
            // compressed_header() の forward update 後の値のまま残す。
            let mut working = starting_probs.clone();
            working.tx_probs = compressed.probs.tx_probs;
            working.skip_prob = compressed.probs.skip_prob;

            let counts = tile_decoder.counts();
            // 仕様 8.4.3 節: updateFactor の決定。
            let update_factor = if header.frame_is_intra {
                112
            } else if self.last_frame_type == Some(FrameType::KeyFrame) {
                128
            } else {
                112
            };
            adapt_coef_probs(&mut working.coef_probs, counts, update_factor);

            if !header.frame_is_intra {
                // load_probs2( frame_context_idx ): tx_probs/skip_prob も forward update 前へ
                // 戻したうえで adapt_noncoef_probs を適用する。
                working.tx_probs = starting_probs.tx_probs;
                working.skip_prob = starting_probs.skip_prob;
                adapt_noncoef_probs(
                    &mut working,
                    counts,
                    header.interpolation_filter,
                    compressed.tx_mode,
                    header.allow_high_precision_mv,
                );
            }
            working
        } else {
            compressed.probs.clone()
        };
        if header.refresh_frame_context {
            self.frame_contexts
                .save(header.frame_context_idx, final_probs);
        }

        // Reference frame update process（仕様 8.10 節）。
        let ref_data = build_ref_frame_data(
            tile_decoder.planes(),
            header.width,
            header.height,
            &header.color_config,
        );
        self.dpb.update(header.refresh_frame_flags, &ref_data);
        for (slot, size) in self.ref_frame_sizes.iter_mut().enumerate() {
            if (header.refresh_frame_flags >> slot) & 1 == 1 {
                *size = (header.width, header.height);
            }
        }

        // compute_image_size は show_existing_frame では呼ばれないため、ここに来た時点で
        // 常に呼ばれたことになる（仕様 7.2.6 節）。UsePrevFrameMvs 判定用に記録する。
        self.prev_frame_dims = Some((header.width, header.height));
        self.prev_show_frame = Some(header.show_frame);
        self.prev_mi_grid = Some(tile_decoder.mi_grid().clone());
        self.loop_filter_deltas = (
            header.loop_filter.ref_deltas,
            header.loop_filter.mode_deltas,
        );
        self.last_frame_type = Some(header.frame_type);

        if header.show_frame {
            Ok(Some(crop_to_frame(
                tile_decoder.planes(),
                header.width,
                header.height,
                &header.color_config,
            )))
        } else {
            Ok(None)
        }
    }
}
