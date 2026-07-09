//! 非圧縮ヘッダ用の生ビットリーダー（仕様 9.1 節 "Parsing process for f(n)"）。
//!
//! VP9 の非圧縮フレームヘッダ（uncompressed_header）は、bool デコーダ（算術符号）ではなく、
//! バイト列を単純にビット単位（MSB が先）で読み出す方式でパースされる。仕様 4.9.1 節に
//! よれば、descriptor が `f(n)` の要素は「ビットストリームに直接現れる符号なし n ビット整数
//! （上位ビットから読む）」であり、`s(n)`（4.9.2 節）は「n ビットの絶対値 + 1 ビットの符号」
//! と定義されている。

/// MSB 優先の生ビットリーダー。
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// 次に読み出すビットの絶対位置（0 = 先頭バイトの最上位ビット）。
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// 新しい `BitReader` を作成する。
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// 仕様 9.1 節の `read_bit()` に相当する。バイト列の範囲外を読もうとした場合は
    /// パニックせず 0 を返す（呼び出し側でストリームの妥当性を別途確認すること）。
    fn read_bit(&mut self) -> u32 {
        let byte_index = self.bit_pos / 8;
        let bit_index_from_msb = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        if byte_index >= self.data.len() {
            return 0;
        }
        ((self.data[byte_index] >> bit_index_from_msb) & 1) as u32
    }

    /// `f(n)` descriptor: 上位ビットから n ビット読み、符号なし整数として返す（仕様 9.1 節）。
    ///
    /// ```text
    /// x = 0
    /// for ( i = 0; i < n; i++ )
    ///     x = 2 * x + read_bit( )
    /// ```
    pub fn f(&mut self, n: u32) -> u32 {
        let mut x = 0u32;
        for _ in 0..n {
            x = 2 * x + self.read_bit();
        }
        x
    }

    /// `s(n)` descriptor: n ビットの絶対値と 1 ビットの符号フラグ（仕様 4.9.2 節）。
    ///
    /// ```text
    /// s(n) {
    ///     value f(n)
    ///     sign f(1)
    ///     return sign ? -value : value
    /// }
    /// ```
    pub fn s(&mut self, n: u32) -> i32 {
        let value = self.f(n) as i32;
        let sign = self.f(1);
        if sign == 1 {
            -value
        } else {
            value
        }
    }

    /// f(1) を bool として読む便利メソッド。
    pub fn flag(&mut self) -> bool {
        self.f(1) == 1
    }

    /// 現在のビット位置（先頭からの絶対ビット数）を返す。
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// 現在のビット位置がバイト境界にあるかどうかを返す。
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos.is_multiple_of(8)
    }

    /// 現在のビット位置を含むバイトまでの、消費済みバイト数を返す
    /// （バイト境界にない場合は切り上げる）。
    pub fn byte_position_ceil(&self) -> usize {
        self.bit_pos.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first() {
        // 0b1011_0010
        let data = [0b1011_0010u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(1), 1);
        assert_eq!(r.f(1), 0);
        assert_eq!(r.f(1), 1);
        assert_eq!(r.f(1), 1);
        assert_eq!(r.f(4), 0b0010);
    }

    #[test]
    fn reads_multi_bit_values_spanning_bytes() {
        // 16 ビット値 0x1234 をビッグエンディアン相当のビット列として読む。
        let data = 0x1234u16.to_be_bytes();
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(16), 0x1234);
    }

    #[test]
    fn reads_signed_value() {
        // value=5 (0b0101), sign=1 -> -5
        let data = [0b0101_1000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.s(4), -5);
    }

    #[test]
    fn reads_signed_value_positive() {
        // value=5 (0b0101), sign=0 -> 5
        let data = [0b0101_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.s(4), 5);
    }

    #[test]
    fn out_of_range_reads_return_zero() {
        let data: [u8; 0] = [];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(8), 0);
    }

    #[test]
    fn byte_position_ceil_rounds_up() {
        let data = [0u8; 4];
        let mut r = BitReader::new(&data);
        let _ = r.f(3);
        assert_eq!(r.byte_position_ceil(), 1);
        let _ = r.f(5);
        assert_eq!(r.byte_position_ceil(), 1);
        let _ = r.f(1);
        assert_eq!(r.byte_position_ceil(), 2);
    }
}
