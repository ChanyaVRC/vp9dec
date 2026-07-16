//! VP9's bool decoder (arithmetic decoder, spec §9.2 "Parsing process for Boolean decoder").
//!
//! Nearly all of the VP9 bitstream except the uncompressed frame header (the
//! compressed header and tile data) is entropy-coded by this bool coder. This
//! implementation follows the spec's pseudocode faithfully and does not reference
//! any existing OSS implementation (clean-room implementation).
//!
//! Spec text referenced (§9.2.1 - §9.2.4):
//!
//! ```text
//! 9.2.1 Initialization process for Boolean decoder
//!   BoolValue = f(8)
//!   BoolRange = 255
//!   BoolMaxBits = 8 * sz - 8
//!   read a marker with read_bool(128), requiring it to be 0
//!
//! 9.2.2 Boolean decoding process (read_bool(p))
//!   split = 1 + (((BoolRange - 1) * p) >> 8)
//!   if BoolValue < split:
//!       BoolRange = split; bool = 0
//!   else:
//!       BoolRange -= split; BoolValue -= split; bool = 1
//!   while BoolRange < 128:
//!       newBit = (BoolMaxBits > 0) ? f(1) : 0   (if read, BoolMaxBits -= 1)
//!       BoolRange *= 2
//!       BoolValue = (BoolValue << 1) + newBit
//!
//! 9.2.3 Exit process for Boolean decoder (exit_bool())
//!   discard the remaining BoolMaxBits worth of padding (expected to be 0)
//!
//! 9.2.4 Parsing process for read_literal(n)
//!   x = 0
//!   for i in 0..n: x = 2 * x + read_bool(128)
//! ```

/// Errors that can occur when initializing the bool decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolCoderError {
    /// `init_bool(sz)` must not be called with sz < 1 (spec §9.2.1).
    EmptyBuffer,
    /// The marker bit read right after initialization must be 0 (spec §9.2.1).
    InvalidMarker,
}

/// VP9's arithmetic decoder (bool decoder).
///
/// `data` is expected to be the `sz`-byte slice from `init_bool( sz )` (the
/// compressed header or tile data itself), passed through as-is.
#[derive(Debug, Clone)]
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    /// Absolute position of the next raw bit to read. `data[0]` is already
    /// consumed as the initial value of BoolValue, so the initial value is 8.
    bit_pos: usize,
    /// BoolValue.
    value: u32,
    /// BoolRange.
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    /// Corresponds to `init_bool( sz )`. `data.len()` is the spec's `sz`.
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
        // In a spec-conformant stream, the marker bit read right after initialization is always 0.
        if decoder.read_bool(128) {
            return Err(BoolCoderError::InvalidMarker);
        }
        Ok(decoder)
    }

    /// Reads 1 bit from the raw bit stream (MSB first). Reading past the end
    /// returns 0, matching the spec's "newBit = 0 when BoolMaxBits == 0" behavior.
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

    /// `read_bool( p )` (spec §9.2.2). Decodes 1 bool based on probability p (0..=255, denominator 256).
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

    /// `read_literal( n )` (spec §9.2.4). Reads n bits with probability 1/2 (p=128), high bit first.
    pub fn read_literal(&mut self, n: u32) -> u32 {
        let mut x = 0u32;
        for _ in 0..n {
            x = 2 * x + self.read_bool(128) as u32;
        }
        x
    }

    /// `read_literal( 1 ) == 1`, as a single-bit flag (spec §9.2.4). A convenience wrapper for
    /// the very common single-bit `L(1)` reads scattered through the compressed header.
    pub fn flag(&mut self) -> bool {
        self.read_literal(1) == 1
    }

    /// Decoding process for tree-coded syntax elements (spec §9.3.3, "Tree decoding process").
    ///
    /// `tree` is the spec's tree array itself (leaves are non-positive values
    /// `-value`, internal nodes are non-negative values pointing to the next
    /// index). `prob_of(node)` is a closure, built by the caller, that returns
    /// the probability obtained from the probability selection process (spec
    /// §9.3.2) given a node number (`n >> 1`).
    ///
    /// ```text
    /// do {
    ///     n = T[ n + read_bool( P( n >> 1 ) ) ]
    /// } while ( n > 0 )
    /// return -n
    /// ```
    pub fn read_tree<F>(&mut self, tree: &[i32], mut prob_of: F) -> i32
    where
        F: FnMut(usize) -> u8,
    {
        let mut n: i32 = 0;
        loop {
            let node = (n as usize) >> 1;
            let bit = self.read_bool(prob_of(node)) as i32;
            n = tree[(n + bit) as usize];
            if n <= 0 {
                break;
            }
        }
        -n
    }

    /// `exit_bool( )` (spec §9.2.3). Discards the remaining padding bits and finishes.
    ///
    /// The spec requires the padding to be 0, but this implementation does not
    /// validate the value and only advances the internal position to the end
    /// of the buffer, so it does not panic on non-conformant streams.
    pub fn exit_bool(&mut self) {
        self.bit_pos = self.data.len() * 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::BoolEncoder;

    /// A simple linear congruential generator (LCG) pseudo-random number generator, for tests only.
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
            // LCG using the Numerical Recipes constants.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 32) as u32
        }

        /// Returns a probability value in 0..=255 (the boundary values 0 and 255 can also occur).
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
        // If the first byte is 128 or more, then with split=128 at BoolRange=255,
        // value(=0xFF) >= split, so marker would read as 1, which is invalid.
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

    /// Round-trip test using random bit sequences x probability sequences. Verified across multiple seeds and lengths.
    #[test]
    fn roundtrip_random_sequences() {
        for seed in [1u64, 2, 42, 1234567, 0xDEAD_BEEF, 999_999_999] {
            for &len in &[0usize, 1, 2, 7, 16, 100, 500, 2000] {
                let mut lcg = Lcg::new(seed ^ len as u64);
                let bits: Vec<bool> = (0..len).map(|_| lcg.next_bool()).collect();
                // Probability 0 is treated as 1 in the split formula (there's a "+1" floor),
                // so the full 0..=255 range can be used as-is.
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
        // Verify that a sequence mixing boundary probability values (0, 1, 254, 255) still round-trips correctly.
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
    fn read_tree_decodes_all_leaves() {
        // A 4-value tree equivalent to PARTITION_TYPES: [ -0, 2, -1, 4, -2, -3 ]
        let tree: [i32; 6] = [0, 2, -1, 4, -2, -3];
        let probs = [100u8, 150u8, 200u8];

        // value 0 -> bit sequence [0]
        // value 1 -> bit sequence [1, 0]
        // value 2 -> bit sequence [1, 1, 0]
        // value 3 -> bit sequence [1, 1, 1]
        let mut enc = BoolEncoder::new();
        enc.write_bool(false, probs[0]); // 0
        enc.write_bool(true, probs[0]);
        enc.write_bool(false, probs[1]); // 1
        enc.write_bool(true, probs[0]);
        enc.write_bool(true, probs[1]);
        enc.write_bool(false, probs[2]); // 2
        enc.write_bool(true, probs[0]);
        enc.write_bool(true, probs[1]);
        enc.write_bool(true, probs[2]); // 3
        let buf = enc.finish();

        let mut dec = BoolDecoder::new(&buf).expect("valid bitstream");
        for expected in [0i32, 1, 2, 3] {
            let got = dec.read_tree(&tree, |node| probs[node]);
            assert_eq!(got, expected);
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
