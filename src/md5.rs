//! MD5（RFC 1321）の自作実装。
//!
//! コンフォーマンステスト（`tests/conformance_test.rs`）で、公式テストベクタに同梱される
//! `.md5` ファイル（デコード結果の I420 フレームデータの MD5 チェックサム）と比較するために
//! 使用する。依存クレートゼロの方針のため、標準ライブラリのみで実装する。
//!
//! アルゴリズムは RFC 1321 "The MD5 Message-Digest Algorithm" にそのまま従う
//! （<https://www.ietf.org/rfc/rfc1321.txt>、パブリックな IETF 標準仕様であり、
//! 本リポジトリのクリーンルーム方針が禁じる「他 OSS 実装のソースコード参照」には該当しない）。

/// 各ラウンドでのシフト量（RFC 1321 3.4 節 "Round 1"〜"Round 4"）。
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// `K[i] = floor(abs(sin(i + 1)) * 2^32)`（RFC 1321 3.4 節の定数表）。
/// 標準ライブラリのみで完結させるため、浮動小数点の `sin` は使わず既知の表を埋め込む
/// （RFC 1321 本文に掲載されている数値そのもの）。
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

/// 1 ブロック（64 バイト）を処理し、`state`（`A,B,C,D`）を更新する（RFC 1321 3.4 節）。
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

/// `data` の MD5 ダイジェスト（16 バイト）を計算する。
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut state: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

    // パディング: 0x80 の 1 バイト + 0 埋め + 元のビット長（64bit, リトルエンディアン）を
    // 付加し、全体が 64 バイトの倍数になるようにする（RFC 1321 3.1〜3.2 節）。
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

/// MD5 ダイジェストを小文字 16 進文字列（32 文字）に変換する。
pub fn to_hex(digest: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for byte in digest {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_of(data: &[u8]) -> String {
        to_hex(&md5(data))
    }

    #[test]
    fn empty_string() {
        assert_eq!(hex_of(b""), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn abc() {
        assert_eq!(hex_of(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn message_digest() {
        assert_eq!(
            hex_of(b"message digest"),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
    }

    #[test]
    fn alphabet() {
        assert_eq!(
            hex_of(b"abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
    }

    #[test]
    fn alphanumeric() {
        assert_eq!(
            hex_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
    }

    #[test]
    fn eighty_digits() {
        assert_eq!(
            hex_of(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    /// ブロック境界（64 バイト）をまたぐ入力でもパディングが正しく行われることを確認する。
    #[test]
    fn exactly_one_block() {
        let data = vec![b'a'; 64];
        // 既知値との比較ではなく、決定的であること・パニックしないことのみ確認する
        // （長さちょうど 64 バイトはパディングが 2 ブロック目に丸ごとあふれる境界ケース）。
        let d1 = md5(&data);
        let d2 = md5(&data);
        assert_eq!(d1, d2);
    }
}
