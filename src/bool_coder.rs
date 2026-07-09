//! VP9 の bool デコーダ（算術復号器、仕様 9.2 節 "Parsing process for Boolean decoder"）。
//!
//! 非圧縮フレームヘッダを除く VP9 ビットストリームのほぼ全て（圧縮ヘッダ・タイルデータ）は、
//! この bool コーダによってエントロピー符号化されている。本実装は仕様の疑似コードに忠実に
//! 実装したものであり、既存 OSS 実装は一切参照していない（クリーンルーム実装）。
//!
//! 参照した仕様の記述（9.2.1 - 9.2.4 節）:
//!
//! ```text
//! 9.2.1 Initialization process for Boolean decoder
//!   BoolValue = f(8)
//!   BoolRange = 255
//!   BoolMaxBits = 8 * sz - 8
//!   read_bool(128) で marker を読み、0 であることを要求する
//!
//! 9.2.2 Boolean decoding process (read_bool(p))
//!   split = 1 + (((BoolRange - 1) * p) >> 8)
//!   if BoolValue < split:
//!       BoolRange = split; bool = 0
//!   else:
//!       BoolRange -= split; BoolValue -= split; bool = 1
//!   while BoolRange < 128:
//!       newBit = (BoolMaxBits > 0) ? f(1) : 0   (読んだ場合 BoolMaxBits -= 1)
//!       BoolRange *= 2
//!       BoolValue = (BoolValue << 1) + newBit
//!
//! 9.2.3 Exit process for Boolean decoder (exit_bool())
//!   残りの BoolMaxBits 分のパディングを読み捨てる（0 であることが期待される）
//!
//! 9.2.4 Parsing process for read_literal(n)
//!   x = 0
//!   for i in 0..n: x = 2 * x + read_bool(128)
//! ```

/// bool デコーダ初期化時に発生し得るエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolCoderError {
    /// `init_bool(sz)` は sz < 1 で呼び出されてはならない（仕様 9.2.1 節）。
    EmptyBuffer,
    /// 初期化直後に読むマーカービットは 0 でなければならない（仕様 9.2.1 節）。
    InvalidMarker,
}

/// VP9 の算術復号器（bool デコーダ）。
///
/// `data` は `init_bool( sz )` の `sz` バイト分のスライス（圧縮ヘッダやタイルデータそのもの）を
/// そのまま渡すことを想定している。
#[derive(Debug, Clone)]
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    /// 次に読み出す生ビットの絶対位置。`data[0]` は BoolValue の初期値として既に消費済みなので
    /// 初期値は 8。
    bit_pos: usize,
    /// BoolValue。
    value: u32,
    /// BoolRange。
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    /// `init_bool( sz )` に相当する。`data.len()` が仕様の `sz` にあたる。
    pub fn new(data: &'a [u8]) -> Result<Self, BoolCoderError> {
        if data.is_empty() {
            return Err(BoolCoderError::EmptyBuffer);
        }
        let mut decoder = Self {
            data,
            bit_pos: 8,
            value: data[0] as u32,
            range: 255,
        };
        // 仕様適合ストリームでは、初期化直後に読まれるマーカービットは常に 0 である。
        if decoder.read_bool(128) {
            return Err(BoolCoderError::InvalidMarker);
        }
        Ok(decoder)
    }

    /// 生のビットストリームから 1 ビット読む（MSB が先）。範囲外を読もうとした場合は
    /// 仕様の「BoolMaxBits == 0 の場合は newBit = 0」に対応する挙動として 0 を返す。
    fn read_bit_raw(&mut self) -> u32 {
        let total_bits = self.data.len() * 8;
        if self.bit_pos >= total_bits {
            return 0;
        }
        let byte_index = self.bit_pos / 8;
        let bit_index_from_msb = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        ((self.data[byte_index] >> bit_index_from_msb) & 1) as u32
    }

    /// `read_bool( p )`（仕様 9.2.2 節）。確率 p (0..=255, 分母 256) に基づいて 1 bool を復号する。
    pub fn read_bool(&mut self, p: u8) -> bool {
        let split = 1 + (((self.range - 1) * p as u32) >> 8);
        let bit = if self.value < split {
            self.range = split;
            false
        } else {
            self.range -= split;
            self.value -= split;
            true
        };
        while self.range < 128 {
            let new_bit = self.read_bit_raw();
            self.range <<= 1;
            self.value = (self.value << 1) + new_bit;
        }
        bit
    }

    /// `read_literal( n )`（仕様 9.2.4 節）。確率 1/2 (p=128) の n ビットを上位ビットから読む。
    pub fn read_literal(&mut self, n: u32) -> u32 {
        let mut x = 0u32;
        for _ in 0..n {
            x = 2 * x + self.read_bool(128) as u32;
        }
        x
    }

    /// `exit_bool( )`（仕様 9.2.3 節）。残りのパディングビットを読み捨てて終了する。
    ///
    /// 仕様上パディングは 0 であることが要求されるが、本実装では非準拠ストリームでも
    /// パニックしないよう、値の検証は行わずバッファ終端まで内部位置を進めるだけに留める。
    pub fn exit_bool(&mut self) {
        self.bit_pos = self.data.len() * 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用の bool エンコーダ。
    ///
    /// [`BoolDecoder`] の逆演算を行う算術符号器で、ラウンドトリップテストのためだけに
    /// このテストモジュール内に実装する（ライブラリ本体には含めない。VP9 はデコーダのみを
    /// 実装対象とするため）。
    ///
    /// 実装方針: このコーダは「範囲 (BoolRange) は 255 を上限に、128 未満になるたびに
    /// 1 ビットずつ倍加して正規化する」という構造を持つため、エンコーダ側は
    /// 「まだ確定していない下位ビットへの桁上げ (キャリー) が、既に出力したビット列に
    /// 遡って影響し得る」という古典的な算術符号化の問題を抱える。
    ///
    /// ここではストリーミングを行わず、エンコード対象の bool 列をすべて受け取ってから
    /// 一括で出力する設計とし、内部状態 `low` を「桁数無制限の 2 進数」として保持することで
    /// キャリー伝搬を厳密な多倍長演算として扱う（近似や特殊なキャリー検出ロジックを
    /// 必要としない）。
    struct BoolEncoder {
        /// `low` の 2 進数表現。`bits[0]` が最上位ビット（最初に確定したビット）、
        /// `bits.last()` が最下位ビット（直近に正規化でシフトインされたビット）。
        /// 先頭 8 ビットは BoolValue が格納される 1 バイト目に対応するプレースホルダーとして
        /// 0 で初期化する。
        low_bits: Vec<u8>,
        /// BoolRange の符号化側での対応物。デコーダと全く同じ更新式で遷移する
        /// （BoolRange の遷移は bool 値と確率だけで決まり、出力ビット列には依存しないため）。
        range: u32,
    }

    impl BoolEncoder {
        fn new() -> Self {
            let mut enc = Self {
                low_bits: vec![0u8; 8], // BoolValue 用の 1 バイト目のプレースホルダー
                range: 255,
            };
            // init_bool の一部として読まれるマーカービット（常に 0）をエンコードする。
            enc.write_bool(false, 128);
            enc
        }

        fn add_split(&mut self, split: u32) {
            // low_bits (最下位ビットが末尾) に対して split を加算し、桁上げを上位へ伝搬する。
            let mut carry = split;
            let mut idx = self.low_bits.len();
            while carry != 0 {
                if idx == 0 {
                    self.low_bits.insert(0, 0);
                    idx = 1;
                }
                idx -= 1;
                let sum = self.low_bits[idx] as u32 + (carry & 1);
                self.low_bits[idx] = (sum & 1) as u8;
                carry = (carry >> 1) + (sum >> 1);
            }
        }

        fn write_bool(&mut self, bit: bool, p: u8) {
            let split = 1 + (((self.range - 1) * p as u32) >> 8);
            if bit {
                self.add_split(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.low_bits.push(0); // 新しい最下位ビットのプレースホルダー（後で桁上げにより 1 になり得る）
                self.range <<= 1;
            }
        }

        fn write_literal(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                let bit = (value >> i) & 1 == 1;
                self.write_bool(bit, 128);
            }
        }

        /// エンコードを終了し、バイト列を返す（`exit_bool` に相当するバイト境界までの
        /// 0 パディングを行う）。
        fn finish(mut self) -> Vec<u8> {
            while !self.low_bits.len().is_multiple_of(8) {
                self.low_bits.push(0);
            }
            self.low_bits
                .chunks(8)
                .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | b))
                .collect()
        }
    }

    /// テスト専用の単純な線形合同法（LCG）による疑似乱数生成器。
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self {
                state: seed ^ 0x9E37_79B9_7F4A_7C15,
            }
        }

        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes の定数を用いた LCG。
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 32) as u32
        }

        /// 0..=255 の確率値を返す（0 と 255 の境界値も出現しうる）。
        fn next_prob(&mut self) -> u8 {
            (self.next_u32() % 256) as u8
        }

        fn next_bool(&mut self) -> bool {
            self.next_u32() % 2 == 1
        }
    }

    #[test]
    fn empty_buffer_is_rejected() {
        let data: [u8; 0] = [];
        assert_eq!(
            BoolDecoder::new(&data).unwrap_err(),
            BoolCoderError::EmptyBuffer
        );
    }

    #[test]
    fn invalid_marker_is_rejected() {
        // 先頭バイトが 128 以上だと、BoolRange=255 での split=128 に対し
        // value(=0xFF) >= split となり marker が 1 と読めてしまうため不正。
        let data = [0xFFu8, 0x00];
        assert_eq!(
            BoolDecoder::new(&data).unwrap_err(),
            BoolCoderError::InvalidMarker
        );
    }

    #[test]
    fn roundtrip_fixed_sequence() {
        let bits = [true, false, false, true, true, true, false, false, true];
        let probs = [128u8, 1, 255, 64, 200, 10, 250, 5, 128];

        let mut enc = BoolEncoder::new();
        for (&b, &p) in bits.iter().zip(probs.iter()) {
            enc.write_bool(b, p);
        }
        let buf = enc.finish();

        let mut dec = BoolDecoder::new(&buf).expect("valid bitstream");
        for (&b, &p) in bits.iter().zip(probs.iter()) {
            assert_eq!(dec.read_bool(p), b);
        }
    }

    #[test]
    fn roundtrip_literal() {
        let mut enc = BoolEncoder::new();
        enc.write_literal(0b1011_0110, 8);
        enc.write_literal(0, 4);
        enc.write_literal(0xF, 4);
        let buf = enc.finish();

        let mut dec = BoolDecoder::new(&buf).expect("valid bitstream");
        assert_eq!(dec.read_literal(8), 0b1011_0110);
        assert_eq!(dec.read_literal(4), 0);
        assert_eq!(dec.read_literal(4), 0xF);
    }

    /// 乱数列 × 確率列によるラウンドトリップテスト。複数のシードと長さで検証する。
    #[test]
    fn roundtrip_random_sequences() {
        for seed in [1u64, 2, 42, 1234567, 0xDEAD_BEEF, 999_999_999] {
            for &len in &[0usize, 1, 2, 7, 16, 100, 500, 2000] {
                let mut lcg = Lcg::new(seed ^ len as u64);
                let bits: Vec<bool> = (0..len).map(|_| lcg.next_bool()).collect();
                // 確率 0 は split の式的には 1 として扱われる（1 + ... の下駄がある）ため
                // 0..=255 全域をそのまま使ってよい。
                let probs: Vec<u8> = (0..len).map(|_| lcg.next_prob()).collect();

                let mut enc = BoolEncoder::new();
                for (&b, &p) in bits.iter().zip(probs.iter()) {
                    enc.write_bool(b, p);
                }
                let buf = enc.finish();

                let mut dec = BoolDecoder::new(&buf)
                    .unwrap_or_else(|e| panic!("seed={seed} len={len}: init failed: {e:?}"));
                for (i, (&b, &p)) in bits.iter().zip(probs.iter()).enumerate() {
                    let got = dec.read_bool(p);
                    assert_eq!(got, b, "seed={seed} len={len} index={i} prob={p}: mismatch");
                }
            }
        }
    }

    #[test]
    fn roundtrip_extreme_probabilities() {
        // 確率の境界値 (0, 1, 254, 255) を混在させたシーケンスでも一致することを確認する。
        let bits = [
            true, true, false, false, true, false, true, false, true, true,
        ];
        let probs = [0u8, 1, 1, 0, 255, 254, 255, 0, 254, 1];

        let mut enc = BoolEncoder::new();
        for (&b, &p) in bits.iter().zip(probs.iter()) {
            enc.write_bool(b, p);
        }
        let buf = enc.finish();

        let mut dec = BoolDecoder::new(&buf).expect("valid bitstream");
        for (&b, &p) in bits.iter().zip(probs.iter()) {
            assert_eq!(dec.read_bool(p), b);
        }
    }

    #[test]
    fn exit_bool_does_not_panic_and_advances_to_end() {
        let mut enc = BoolEncoder::new();
        enc.write_literal(5, 4);
        let buf = enc.finish();

        let mut dec = BoolDecoder::new(&buf).expect("valid bitstream");
        let _ = dec.read_literal(4);
        dec.exit_bool();
        assert_eq!(dec.bit_pos, buf.len() * 8);
    }
}
