//! Raw bit reader for the uncompressed header (spec §9.1, "Parsing process for f(n)").
//!
//! VP9's uncompressed frame header (uncompressed_header) is parsed not by the bool
//! decoder (arithmetic coding) but by simply reading the byte stream bit by bit
//! (MSB first). Per spec §4.9.1, a descriptor of `f(n)` is "an unsigned n-bit
//! integer that appears directly in the bitstream (read from the high bit)",
//! while `s(n)` (§4.9.2) is defined as "an n-bit absolute value plus a 1-bit sign".

/// MSB-first raw bit reader.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute position of the next bit to read (0 = the MSB of the first byte).
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Creates a new `BitReader`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Corresponds to `read_bit()` in spec §9.1. Reading past the end of the byte
    /// slice does not panic and returns 0 instead (callers must otherwise verify
    /// stream validity).
    fn read_bit(&mut self) -> u32 {
        let byte_index = self.bit_pos / 8;
        let bit_index_from_msb = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        if byte_index >= self.data.len() {
            return 0;
        }
        ((self.data[byte_index] >> bit_index_from_msb) & 1) as u32
    }

    /// `f(n)` descriptor: reads n bits starting from the high bit and returns them
    /// as an unsigned integer (spec §9.1).
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

    /// `s(n)` descriptor: an n-bit absolute value plus a 1-bit sign flag (spec §4.9.2).
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

    /// Convenience method that reads f(1) as a bool.
    pub fn flag(&mut self) -> bool {
        self.f(1) == 1
    }

    /// Returns the current bit position (absolute bit count from the start).
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Returns whether the current bit position lies on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos.is_multiple_of(8)
    }

    /// Returns the number of bytes consumed up to and including the byte that
    /// holds the current bit position (rounded up if not on a byte boundary).
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
        // Read the 16-bit value 0x1234 as a bit stream equivalent to big-endian.
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
