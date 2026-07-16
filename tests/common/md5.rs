//! A from-scratch MD5 (RFC 1321) implementation.
//!
//! Used by the conformance test (`tests/conformance_test.rs`) to compare against the
//! `.md5` files bundled with the official test vectors (MD5 checksums of the decoded
//! I420 frame data). Lives under `tests/common/` (relocated from `src/md5.rs` in Wave 3,
//! 2026-07-16) since it has no consumer outside the test suite; implemented using only the
//! standard library, per the zero dependency crates policy.
//!
//! The algorithm follows RFC 1321 "The MD5 Message-Digest Algorithm" directly
//! (<https://www.ietf.org/rfc/rfc1321.txt>, a public IETF standard specification,
//! which does not fall under this repository's clean-room policy's prohibition on
//! "referencing other OSS implementations' source code").
//!
//! Unit tests for this module live in `tests/conformance_test.rs`'s `mod md5_tests` rather
//! than here: everything under `tests/common/` is recompiled once per consuming test binary,
//! so a `#[test]` here would rerun once per binary instead of once overall.

/// The shift amount for each round (RFC 1321 §3.4 "Round 1" through "Round 4").
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// `K[i] = floor(abs(sin(i + 1)) * 2^32)` (the constant table from RFC 1321 §3.4).
/// To stay within the standard library, this embeds the known table rather than
/// computing floating-point `sin` (the values are exactly as listed in RFC 1321).
const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// Processes one block (64 bytes) and updates `state` (`A,B,C,D`) (RFC 1321 §3.4).
fn process_block(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut m = [0u32; 16];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        m[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    let [mut a, mut b, mut c, mut d] = *state;

    for i in 0..64 {
        let (f, g) = match i {
            0..=15 => ((b & c) | (!b & d), i),
            16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
        a = d;
        d = c;
        c = b;
        b = b.wrapping_add(f.rotate_left(SHIFTS[i]));
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// Computes the MD5 digest (16 bytes) of `data`.
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut state: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

    // Padding: append a single 0x80 byte + zero padding + the original bit length
    // (64-bit, little-endian) so the total length becomes a multiple of 64 bytes
    // (RFC 1321 §3.1-§3.2).
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_le_bytes());
    debug_assert_eq!(padded.len() % 64, 0);

    for block in padded.chunks_exact(64) {
        let block: &[u8; 64] = block.try_into().unwrap();
        process_block(&mut state, block);
    }

    let mut out = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// Converts an MD5 digest to a lowercase hex string (32 characters).
pub fn to_hex(digest: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for byte in digest {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}
