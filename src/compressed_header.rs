//! 圧縮ヘッダ（`compressed_header`）のパース（仕様 6.3 節）。
//!
//! `compressed_header()` は `header_size_in_bytes` バイト分の bool 符号化されたデータで、
//! 変換モード (`tx_mode`) と各種確率テーブルの更新内容を保持する。
//!
//! ```text
//! compressed_header( ) {
//!     read_tx_mode( )
//!     if ( tx_mode == TX_MODE_SELECT ) {
//!        tx_mode_probs( )
//!     }
//!     read_coef_probs( )
//!     read_skip_prob( )
//!     if ( FrameIsIntra == 0 ) {
//!        read_inter_mode_probs( )
//!        ...
//!        mv_probs( )
//!     }
//! }
//! ```
//!
//! `FrameIsIntra == 0` の場合にのみ呼ばれるインター関連の読み取り（`read_inter_mode_probs`
//! 以降）は、キーフレーム（`FrameIsIntra == 1`）では仕様上そもそも呼び出されない。
//! つまり「読み飛ばす」処理ではなく「存在しない」ため、本実装では何もしない
//! （M3 でインターフレームに対応する際に追加する）。

use crate::bool_coder::{BoolCoderError, BoolDecoder};
use crate::prob_tables::{
    CoefProbs, DEFAULT_COEF_PROBS, DEFAULT_SKIP_PROB, DEFAULT_TX_PROBS, INV_MAP_TABLE, TX_16X16,
    TX_32X32, TX_4X4, TX_8X8, TX_MODE_SELECT, TX_MODE_TO_BIGGEST_TX_SIZE,
};

/// `compressed_header` パース時に発生し得るエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressedHeaderError {
    /// bool デコーダの初期化に失敗した（`header_size_in_bytes` が 0 など）。
    BoolCoder(BoolCoderError),
}

/// `compressed_header()` で更新される確率テーブル一式（キーフレームで使用する範囲のみ）。
///
/// インター予測関連の確率テーブル（`inter_mode_probs`、`mv_probs` 等）は
/// `FrameIsIntra == 1` では読まれないため、ここには含めていない（M3 で追加する）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedHeaderProbs {
    /// `tx_probs[maxTxSize][ctx][node]`。[`crate::prob_tables::DEFAULT_TX_PROBS`] と同じレイアウト。
    pub tx_probs: [[[u8; 3]; 2]; 4],
    /// `coef_probs[txSz][plane>0][is_inter][band][ctx][node]`。
    pub coef_probs: CoefProbs,
    /// `skip_prob[ctx]`（仕様 6.3.8 節）。
    pub skip_prob: [u8; 3],
}

impl Default for CompressedHeaderProbs {
    fn default() -> Self {
        Self {
            tx_probs: DEFAULT_TX_PROBS,
            coef_probs: DEFAULT_COEF_PROBS,
            skip_prob: DEFAULT_SKIP_PROB,
        }
    }
}

/// `compressed_header()` のパース結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedHeader {
    /// `tx_mode`（仕様 7.3.1 節）。
    pub tx_mode: u8,
    /// 更新後の確率テーブル一式。
    pub probs: CompressedHeaderProbs,
}

/// `read_prob()` に相当する `B(252)` の bool 読み取り + 更新判定（仕様 6.3.3 節 `diff_update_prob`）。
///
/// ```text
/// diff_update_prob( prob ) {
///     update_prob                  B(252)
///     if ( update_prob == 1 ) {
///        deltaProb = decode_term_subexp( )
///        prob = inv_remap_prob( deltaProb, prob )
///     }
///     return prob
/// }
/// ```
fn diff_update_prob(r: &mut BoolDecoder, prob: u8) -> u8 {
    let update_prob = r.read_bool(252);
    if update_prob {
        let delta_prob = decode_term_subexp(r);
        inv_remap_prob(delta_prob, prob)
    } else {
        prob
    }
}

/// `decode_term_subexp()`（仕様 6.3.4 節）。すべてのフィールドは `L(n)`（`read_literal`）で読む。
fn decode_term_subexp(r: &mut BoolDecoder) -> u32 {
    if r.read_literal(1) == 0 {
        return r.read_literal(4);
    }
    if r.read_literal(1) == 0 {
        return r.read_literal(4) + 16;
    }
    if r.read_literal(1) == 0 {
        return r.read_literal(5) + 32;
    }
    let v = r.read_literal(7);
    if v < 65 {
        return v + 64;
    }
    let bit = r.read_literal(1);
    (v << 1) - 1 + bit
}

/// `inv_remap_prob( deltaProb, prob )`（仕様 6.3.5 節）。
fn inv_remap_prob(delta_prob: u32, prob: u8) -> u8 {
    let v = INV_MAP_TABLE[delta_prob as usize] as u32;
    // m--（prob から 1 引いた値）を以降で使う。
    let m = prob as i32 - 1;
    let result = if (m << 1) <= 255 {
        1 + inv_recenter_nonneg(v, m as u32) as i32
    } else {
        255 - inv_recenter_nonneg(v, (255 - 1 - m) as u32) as i32
    };
    result as u8
}

/// `inv_recenter_nonneg( v, m )`（仕様 6.3.6 節）。
fn inv_recenter_nonneg(v: u32, m: u32) -> u32 {
    if v > 2 * m {
        return v;
    }
    if v & 1 == 1 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// `read_tx_mode()`（仕様 6.3.1 節）。
fn read_tx_mode(r: &mut BoolDecoder, lossless: bool) -> u8 {
    if lossless {
        TX_4X4 // ONLY_4X4 と同じ値 (0)
    } else {
        let mut tx_mode = r.read_literal(2) as u8;
        if tx_mode == TX_32X32 {
            // ALLOW_32X32 (=3) と TX_32X32 (=3) は値が一致するため同じ定数を使い回している。
            let tx_mode_select = r.read_literal(1) as u8;
            tx_mode += tx_mode_select;
        }
        tx_mode
    }
}

/// `tx_mode_probs()`（仕様 6.3.2 節）。`tx_mode == TX_MODE_SELECT` の場合のみ呼ばれる。
fn read_tx_mode_probs(r: &mut BoolDecoder, tx_probs: &mut [[[u8; 3]; 2]; 4]) {
    // tx_probs_8x8[ TX_SIZE_CONTEXTS ][ TX_SIZES - 3 = 1 ]
    for ctx in tx_probs[TX_8X8 as usize].iter_mut() {
        for node in ctx.iter_mut().take(1) {
            *node = diff_update_prob(r, *node);
        }
    }
    // tx_probs_16x16[ TX_SIZE_CONTEXTS ][ TX_SIZES - 2 = 2 ]
    for ctx in tx_probs[TX_16X16 as usize].iter_mut() {
        for node in ctx.iter_mut().take(2) {
            *node = diff_update_prob(r, *node);
        }
    }
    // tx_probs_32x32[ TX_SIZE_CONTEXTS ][ TX_SIZES - 1 = 3 ]
    for ctx in tx_probs[TX_32X32 as usize].iter_mut() {
        for node in ctx.iter_mut().take(3) {
            *node = diff_update_prob(r, *node);
        }
    }
}

/// `read_coef_probs()`（仕様 6.3.7 節）。
fn read_coef_probs(r: &mut BoolDecoder, tx_mode: u8, coef_probs: &mut CoefProbs) {
    let max_tx_size = TX_MODE_TO_BIGGEST_TX_SIZE[tx_mode as usize];
    for tx_sz in 0..=max_tx_size {
        let update_probs = r.read_literal(1) == 1;
        if !update_probs {
            continue;
        }
        for plane_probs in coef_probs[tx_sz as usize].iter_mut() {
            for ref_probs in plane_probs.iter_mut() {
                for (k, band_probs) in ref_probs.iter_mut().enumerate() {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for ctx_probs in band_probs.iter_mut().take(max_l) {
                        for prob in ctx_probs.iter_mut() {
                            *prob = diff_update_prob(r, *prob);
                        }
                    }
                }
            }
        }
    }
}

/// `read_skip_prob()`（仕様 6.3.8 節）。
fn read_skip_prob(r: &mut BoolDecoder, skip_prob: &mut [u8; 3]) {
    for prob in skip_prob.iter_mut() {
        *prob = diff_update_prob(r, *prob);
    }
}

/// `compressed_header()`（仕様 6.3 節）をパースする。
///
/// `data` は `header_size_in_bytes` バイト分のスライス。`lossless` は非圧縮ヘッダの
/// `quantization_params()` から得られる `Lossless` フラグ。
///
/// 本関数はキーフレーム（`FrameIsIntra == 1`）のみを対象とする。インター予測関連の
/// フィールドは仕様上 `FrameIsIntra == 0` でのみ読まれるため、ここでは読まない。
pub fn parse_compressed_header(
    data: &[u8],
    lossless: bool,
) -> Result<CompressedHeader, CompressedHeaderError> {
    let mut r = BoolDecoder::new(data).map_err(CompressedHeaderError::BoolCoder)?;
    let mut probs = CompressedHeaderProbs::default();

    let tx_mode = read_tx_mode(&mut r, lossless);
    if tx_mode == TX_MODE_SELECT {
        read_tx_mode_probs(&mut r, &mut probs.tx_probs);
    }
    read_coef_probs(&mut r, tx_mode, &mut probs.coef_probs);
    read_skip_prob(&mut r, &mut probs.skip_prob);

    r.exit_bool();

    Ok(CompressedHeader { tx_mode, probs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::test_support::BoolEncoder;
    use crate::prob_tables::ONLY_4X4;

    #[test]
    fn lossless_frame_forces_only_4x4_and_reads_no_extra_bit() {
        // lossless == true の場合 tx_mode は常に ONLY_4X4 で、ビットストリームからは
        // 何も読まない（read_tx_mode の if 分岐が丸ごとスキップされる）ため、
        // 直後の read_coef_probs (txSz は TX_4X4 のみ) と read_skip_prob だけをエンコードする。
        let mut enc = BoolEncoder::new();
        enc.write_literal(0, 1); // read_coef_probs: txSz=TX_4X4, update_probs=0
        enc.write_bool(false, 252); // read_skip_prob[0]: update_prob=0
        enc.write_bool(false, 252); // read_skip_prob[1]
        enc.write_bool(false, 252); // read_skip_prob[2]
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, true).expect("should parse");
        assert_eq!(header.tx_mode, ONLY_4X4);
        assert_eq!(header.probs, CompressedHeaderProbs::default());
    }

    #[test]
    fn non_lossless_reads_two_bit_tx_mode_without_select() {
        // tx_mode = ALLOW_16X16 (=2) を読み、ALLOW_32X32 でないため追加ビットは読まない。
        // maxTxSize = TX_16X16 なので read_coef_probs は TX_4X4, TX_8X8, TX_16X16 の 3 回。
        let mut enc = BoolEncoder::new();
        enc.write_literal(2, 2); // tx_mode = ALLOW_16X16
        enc.write_literal(0, 1); // txSz=TX_4X4 update_probs=0
        enc.write_literal(0, 1); // txSz=TX_8X8 update_probs=0
        enc.write_literal(0, 1); // txSz=TX_16X16 update_probs=0
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, false).expect("should parse");
        assert_eq!(header.tx_mode, 2); // ALLOW_16X16
        assert_eq!(header.probs, CompressedHeaderProbs::default());
    }

    #[test]
    fn tx_mode_select_reads_tx_mode_probs_and_full_coef_range() {
        // tx_mode = ALLOW_32X32 (=3) + tx_mode_select(1) = TX_MODE_SELECT (=4)
        let mut enc = BoolEncoder::new();
        enc.write_literal(3, 2); // tx_mode raw = ALLOW_32X32
        enc.write_literal(1, 1); // tx_mode_select = 1 -> tx_mode = TX_MODE_SELECT
                                 // tx_mode_probs(): 8x8(2*1) + 16x16(2*2) + 32x32(2*3) = 12 回の diff_update_prob
        for _ in 0..12 {
            enc.write_bool(false, 252);
        }
        // read_coef_probs: maxTxSize = TX_32X32 なので txSz = 0..=3 の 4 回
        for _ in 0..4 {
            enc.write_literal(0, 1);
        }
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, false).expect("should parse");
        assert_eq!(header.tx_mode, TX_MODE_SELECT);
        assert_eq!(header.probs, CompressedHeaderProbs::default());
    }

    #[test]
    fn diff_update_prob_actually_changes_skip_prob() {
        // skip_prob[0] を更新する: update_prob=1, decode_term_subexp() の最初の分岐
        // (bit=0 -> sub_exp_val (L(4))) で deltaProb=5 を得て、inv_remap_prob(5, 192) を適用する。
        let mut enc = BoolEncoder::new();
        enc.write_literal(0, 1); // read_coef_probs: txSz=TX_4X4, update_probs=0
        enc.write_bool(true, 252); // skip_prob[0]: update_prob=1
        enc.write_literal(0, 1); // decode_term_subexp: bit=0
        enc.write_literal(5, 4); // sub_exp_val = 5
        enc.write_bool(false, 252); // skip_prob[1]: update_prob=0
        enc.write_bool(false, 252); // skip_prob[2]: update_prob=0
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, true).expect("should parse");
        let expected = inv_remap_prob(5, DEFAULT_SKIP_PROB[0]);
        assert_eq!(header.probs.skip_prob[0], expected);
        assert_ne!(header.probs.skip_prob[0], DEFAULT_SKIP_PROB[0]);
        assert_eq!(header.probs.skip_prob[1], DEFAULT_SKIP_PROB[1]);
        assert_eq!(header.probs.skip_prob[2], DEFAULT_SKIP_PROB[2]);
    }

    #[test]
    fn inv_recenter_nonneg_matches_spec_cases() {
        // v > 2m -> v をそのまま返す
        assert_eq!(inv_recenter_nonneg(10, 3), 10);
        // v が奇数 -> m - (v+1)/2
        assert_eq!(inv_recenter_nonneg(3, 5), 5 - 2);
        // v が偶数 -> m + v/2
        assert_eq!(inv_recenter_nonneg(4, 5), 5 + 2);
    }

    #[test]
    fn empty_data_is_rejected() {
        let data: [u8; 0] = [];
        assert_eq!(
            parse_compressed_header(&data, true).unwrap_err(),
            CompressedHeaderError::BoolCoder(BoolCoderError::EmptyBuffer)
        );
    }
}
