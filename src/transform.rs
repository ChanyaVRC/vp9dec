//! 逆変換（Inverse Transform）モジュール。
//!
//! 参照仕様: VP9 Bitstream & Decoding Process Specification v0.7, 8.7 節
//! "Inverse transform process"。
//!
//! 本モジュールは以下を提供する:
//! - 逆 DCT（4x4 / 8x8 / 16x16 / 32x32）: 仕様 8.7.1.2〜8.7.1.3 節の
//!   permutation + butterfly ネットワークをそのまま実装した `idct4/8/16/32`。
//! - 逆 ADST（4x4 / 8x8 / 16x16）: 仕様 8.7.1.4〜8.7.1.9 節の `iadst4/8/16`。
//! - 逆 Walsh-Hadamard 変換（4x4, ロスレス用）: 仕様 8.7.1.10 節の `iwht4`。
//! - 2 次元変換の組み立て（行変換 → 列変換 → 最終シフト）: 仕様 8.7.2 節の
//!   `inverse_transform_block`。
//!
//! 変換の中間値は仕様が要求する精度（`8 + BitDepth` bit、ADST の `S` 配列は
//! `24 + BitDepth` bit）を余裕を持って収められるよう、すべて `i64` で計算する。

/// 仕様 4.6 節 `Round2(x, n) = (x + (1 << (n-1))) >> n`。
///
/// 本モジュール内での呼び出しは常に `n >= 1` である（仕様上 `n == 0` での
/// 呼び出しは発生しない）。
#[inline]
fn round2(x: i64, n: u32) -> i64 {
    debug_assert!(n >= 1);
    (x + (1i64 << (n - 1))) >> n
}

/// 仕様 8.7.1.1 節 `cos64_lookup[ 33 ]`。
/// `cos64_lookup[i] == round(16384 * cos(i * pi / 64))`。
#[rustfmt::skip]
const COS64_LOOKUP: [i64; 33] = [
    16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426,
    15137, 14811, 14449, 14053, 13623, 13160, 12665, 12140,
    11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005,
    6270, 5520, 4756, 3981, 3196, 2404, 1606, 804,
    0,
];

/// 仕様 8.7.1.1 節 `cos64( angle )`。
fn cos64(angle: i32) -> i64 {
    // `angle & 127` は 2 の補数表現において `angle.rem_euclid(128)` と等価。
    let angle2 = angle.rem_euclid(128);
    match angle2 {
        0..=32 => COS64_LOOKUP[angle2 as usize],
        33..=64 => -COS64_LOOKUP[(64 - angle2) as usize],
        65..=96 => -COS64_LOOKUP[(angle2 - 64) as usize],
        _ => COS64_LOOKUP[(128 - angle2) as usize],
    }
}

/// 仕様 8.7.1.1 節 `sin64( angle ) = cos64( angle - 32 )`。
fn sin64(angle: i32) -> i64 {
    cos64(angle - 32)
}

/// 仕様 8.7.1.1 節 `brev(numBits, x)`（ビット反転）。
fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0usize;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

/// 仕様 8.7.1.1 節 `B( a, b, angle, flip )` バタフライ回転。
fn b_op(t: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    let ta = t[a];
    let tb = t[b];
    let x = ta * cos64(angle) - tb * sin64(angle);
    let y = ta * sin64(angle) + tb * cos64(angle);
    t[a] = round2(x, 14);
    t[b] = round2(y, 14);
    if flip {
        t.swap(a, b);
    }
}

/// 仕様 8.7.1.1 節 `H( a, b, flip )` アダマール回転。
fn h_op(t: &mut [i64], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = x + y;
    t[b] = x - y;
}

/// 仕様 8.7.1.1 節 `SB( a, b, angle, flip )`（高精度 `S` 配列への回転）。
fn sb_op(s: &mut [i64], t: &[i64], a: usize, b: usize, angle: i32, flip: bool) {
    let ta = t[a];
    let tb = t[b];
    let sa = ta * cos64(angle) - tb * sin64(angle);
    let sb = ta * sin64(angle) + tb * cos64(angle);
    if flip {
        s[a] = sb;
        s[b] = sa;
    } else {
        s[a] = sa;
        s[b] = sb;
    }
}

/// 仕様 8.7.1.1 節 `SH( a, b )`。
fn sh_op(t: &mut [i64], s: &[i64], a: usize, b: usize) {
    t[a] = round2(s[a] + s[b], 14);
    t[b] = round2(s[a] - s[b], 14);
}

/// 仕様 8.7.1.2 節: 逆 DCT 用の入力配列並べ替え（ビット反転permutation）。
fn idct_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let copy_t: Vec<i64> = t[..n0].to_vec();
    for i in 0..n0 {
        t[i] = copy_t[brev(n, i)];
    }
}

/// 仕様 8.7.1.3 節: 逆 DCT バタフライネットワーク本体（`2 <= n <= 5`）。
///
/// 呼び出し前に [`idct_permute`] 済みであることが前提。
fn idct(t: &mut [i64], n: u32) {
    let n0 = 1i64 << n;
    let n1 = 1i64 << (n - 1);
    let n2 = 1i64 << (n - 2);

    // 1. n==2 なら直接バタフライ、そうでなければ n-1 で再帰（前半 n1 要素のみを使う）。
    if n == 2 {
        b_op(t, 0, 1, 16, true);
    } else {
        idct(t, n - 1);
    }

    // 2. Invoke B( n1+i, n0-1-i, 32-brev(5, n1+i), 0 ) for i = 0..(n2-1).
    for i in 0..n2 {
        let a = (n1 + i) as usize;
        let b = (n0 - 1 - i) as usize;
        let angle = 32 - brev(5, a) as i32;
        b_op(t, a, b, angle, false);
    }

    // 3. n>=3: Invoke H( n1+4*i+2*j, n1+1+4*i+2*j, j ) for i = 0..(n3-1), j = 0..1.
    if n >= 3 {
        let n3 = 1i64 << (n - 3);
        for i in 0..n3 {
            for j in 0..2i64 {
                let a = (n1 + 4 * i + 2 * j) as usize;
                let b = (n1 + 1 + 4 * i + 2 * j) as usize;
                h_op(t, a, b, j == 1);
            }
        }
    }

    // 4. n==5 の追加ステージ。
    if n == 5 {
        for i in 0..2i64 {
            for j in 0..2i64 {
                let a = (n0 - n as i64 + 3 - n2 * j - 4 * i) as usize;
                let b = (n1 + n as i64 - 4 + n2 * j + 4 * i) as usize;
                let angle = 28 - 16 * i as i32 + 56 * j as i32;
                b_op(t, a, b, angle, true);
            }
        }
        let n3 = 1i64 << (n - 3);
        for i in 0..2i64 {
            for j in 0..4i64 {
                let a = (n1 + n3 * j + i) as usize;
                let b = (n1 + n2 - 5 + n3 * j - i) as usize;
                h_op(t, a, b, (j & 1) == 1);
            }
        }
    }

    // 5. n>=4 の追加ステージ。
    if n >= 4 {
        let imax_a: i64 = if n == 5 { 1 } else { 0 };
        for i in 0..=imax_a {
            for j in 0..2i64 {
                let a = (n0 - n as i64 + 2 - i - n2 * j) as usize;
                let b = (n1 + n as i64 - 3 + i + n2 * j) as usize;
                let angle = 24 + 48 * j as i32;
                b_op(t, a, b, angle, true);
            }
        }
        let imax_b: i64 = 2 * n as i64 - 7;
        for j in 0..2i64 {
            for i in 0..=imax_b {
                let a = (n1 + n2 * j + i) as usize;
                let b = (n1 + n2 - 1 + n2 * j - i) as usize;
                h_op(t, a, b, (j & 1) == 1);
            }
        }
    }

    // 6. n>=3: Invoke B( n0-n3-1-i, n1+n3+i, 16, 1 ) for i = 0..(n3-1).
    if n >= 3 {
        let n3 = 1i64 << (n - 3);
        for i in 0..n3 {
            let a = (n0 - n3 - 1 - i) as usize;
            let b = (n1 + n3 + i) as usize;
            b_op(t, a, b, 16, true);
        }
    }

    // 7. Invoke H( i, n0-1-i, 0 ) for i = 0..(n1-1).
    for i in 0..n1 {
        let a = i as usize;
        let b = (n0 - 1 - i) as usize;
        h_op(t, a, b, false);
    }
}

/// 仕様 8.7.1.4 節: 逆 ADST 入力配列並べ替え。
fn adst_input_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let copy_t: Vec<i64> = t[..n0].to_vec();
    for i in 0..n1 {
        t[2 * i] = copy_t[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy_t[2 * i];
    }
}

/// 仕様 8.7.1.5 節: 逆 ADST 出力配列並べ替え（`n` は 3 または 4）。
fn adst_output_permute(t: &mut [i64], n: u32) {
    if n == 4 {
        let copy_t: Vec<i64> = t[..16].to_vec();
        for a in 0..2usize {
            for b in 0..2usize {
                for c in 0..2usize {
                    for d in 0..2usize {
                        let dst = 8 * a + 4 * b + 2 * c + d;
                        let src = 8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a;
                        t[dst] = copy_t[src];
                    }
                }
            }
        }
    } else {
        debug_assert_eq!(n, 3);
        let copy_t: Vec<i64> = t[..8].to_vec();
        for a in 0..2usize {
            for b in 0..2usize {
                for c in 0..2usize {
                    let dst = 4 * a + 2 * b + c;
                    let src = 4 * (c ^ b) + 2 * (b ^ a) + a;
                    t[dst] = copy_t[src];
                }
            }
        }
    }
}

/// 仕様 8.7.1.6 節の定数 `SINPI_k_9`。
const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

/// 仕様 8.7.1.6 節: 逆 ADST4 本体。
fn iadst4_impl(t: &mut [i64]) {
    let s0 = SINPI_1_9 * t[0];
    let s1 = SINPI_2_9 * t[0];
    let s2 = SINPI_3_9 * t[1];
    let s3 = SINPI_4_9 * t[2];
    let s4 = SINPI_1_9 * t[2];
    let s5 = SINPI_2_9 * t[3];
    let s6 = SINPI_4_9 * t[3];
    let v = t[0] - t[2] + t[3];
    let s7 = SINPI_3_9 * v;
    let x0 = s0 + s3 + s5;
    let x1 = s1 - s4 - s6;
    let x2 = s7;
    let x3 = s2;
    let o0 = x0 + x3;
    let o1 = x1 + x3;
    let o2 = x2;
    let o3 = x0 + x1 - x3;
    t[0] = round2(o0, 14);
    t[1] = round2(o1, 14);
    t[2] = round2(o2, 14);
    t[3] = round2(o3, 14);
}

/// 仕様 8.7.1.7 節: 逆 ADST8 本体。
fn iadst8_impl(t: &mut [i64]) {
    adst_input_permute(t, 3);
    let mut s = [0i64; 8];

    for i in 0..4usize {
        sb_op(&mut s, t, 2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
    }
    for i in 0..4usize {
        sh_op(t, &s, i, 4 + i);
    }
    for i in 0..2usize {
        sb_op(&mut s, t, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
    }
    for i in 0..2usize {
        sh_op(t, &s, 4 + i, 6 + i);
    }
    for i in 0..2usize {
        h_op(t, i, 2 + i, false);
    }
    for i in 0..2usize {
        b_op(t, 2 + 4 * i, 3 + 4 * i, 16, true);
    }

    adst_output_permute(t, 3);

    for i in 0..4usize {
        t[1 + 2 * i] = -t[1 + 2 * i];
    }
}

/// 仕様 8.7.1.8 節: 逆 ADST16 本体。
fn iadst16_impl(t: &mut [i64]) {
    adst_input_permute(t, 4);
    let mut s = [0i64; 16];

    for i in 0..8usize {
        sb_op(&mut s, t, 2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
    }
    for i in 0..8usize {
        sh_op(t, &s, i, 8 + i);
    }
    for i in 0..4usize {
        sb_op(&mut s, t, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    for i in 0..4usize {
        sh_op(t, &s, 8 + i, 12 + i);
    }
    for i in 0..4usize {
        h_op(t, i, 4 + i, false);
    }
    for i in 0..2usize {
        for j in 0..2usize {
            sb_op(
                &mut s,
                t,
                4 + 8 * i + 3 * j,
                5 + 8 * i + j,
                24 - 16 * j as i32,
                true,
            );
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            sh_op(t, &s, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            h_op(t, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            let angle = 48 + 64 * (i ^ j) as i32;
            b_op(t, 2 + 4 * j + 8 * i, 3 + 4 * j + 8 * i, angle, false);
        }
    }

    adst_output_permute(t, 4);

    for i in 0..2usize {
        for j in 0..2usize {
            t[1 + 12 * j + 2 * i] = -t[1 + 12 * j + 2 * i];
        }
    }
}

/// 仕様 8.7.1.9 節: `n` に応じて ADST4/8/16 を選択する（内部ディスパッチ用）。
fn iadst(t: &mut [i64], n: u32) {
    match n {
        2 => iadst4_impl(t),
        3 => iadst8_impl(t),
        4 => iadst16_impl(t),
        _ => unreachable!("iadst は n = 2..=4 のみサポートする"),
    }
}

/// 仕様 8.7.1.2〜8.7.1.3 節: 逆 DCT（サイズ 4）。
pub fn idct4(t: &mut [i64; 4]) {
    idct_permute(&mut t[..], 2);
    idct(&mut t[..], 2);
}

/// 仕様 8.7.1.2〜8.7.1.3 節: 逆 DCT（サイズ 8）。
pub fn idct8(t: &mut [i64; 8]) {
    idct_permute(&mut t[..], 3);
    idct(&mut t[..], 3);
}

/// 仕様 8.7.1.2〜8.7.1.3 節: 逆 DCT（サイズ 16）。
pub fn idct16(t: &mut [i64; 16]) {
    idct_permute(&mut t[..], 4);
    idct(&mut t[..], 4);
}

/// 仕様 8.7.1.2〜8.7.1.3 節: 逆 DCT（サイズ 32）。
pub fn idct32(t: &mut [i64; 32]) {
    idct_permute(&mut t[..], 5);
    idct(&mut t[..], 5);
}

/// 仕様 8.7.1.4〜8.7.1.6 節: 逆 ADST（サイズ 4）。
pub fn iadst4(t: &mut [i64; 4]) {
    iadst4_impl(&mut t[..]);
}

/// 仕様 8.7.1.4〜8.7.1.5、8.7.1.7 節: 逆 ADST（サイズ 8）。
pub fn iadst8(t: &mut [i64; 8]) {
    iadst8_impl(&mut t[..]);
}

/// 仕様 8.7.1.4〜8.7.1.5、8.7.1.8 節: 逆 ADST（サイズ 16）。
pub fn iadst16(t: &mut [i64; 16]) {
    iadst16_impl(&mut t[..]);
}

/// 仕様 8.7.1.10 節: 逆 Walsh-Hadamard 変換（ロスレス、4x4 専用）。
///
/// `shift` は行変換時 2、列変換時 0 を指定する（仕様 8.7.2 節）。
pub fn iwht4(t: &mut [i64; 4], shift: u32) {
    let mut a = t[0] >> shift;
    let mut c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

/// 仕様 8.7.2 節 `TxType`（列方向_行方向の順で命名）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxType {
    /// 列: DCT / 行: DCT
    DctDct,
    /// 列: ADST / 行: DCT
    AdstDct,
    /// 列: DCT / 行: ADST
    DctAdst,
    /// 列: ADST / 行: ADST
    AdstAdst,
}

/// 仕様 8.7.2 節: 2 次元逆変換。
///
/// `dequant` は行優先（row-major）で格納された `n0 x n0`（`n0 = 1 << n`）の
/// 逆量子化済み係数配列であり、変換後の結果もこの配列に上書きされる。
/// `lossless` が真の場合は `tx_type` に関わらず逆 WHT（`n == 2` 専用）を用いる。
///
/// # Panics
/// `dequant.len() != (1 << n) * (1 << n)` の場合、または `lossless == true` かつ
/// `n != 2` の場合にパニックする。
pub fn inverse_transform_block(dequant: &mut [i64], n: u32, tx_type: TxType, lossless: bool) {
    let n0 = 1usize << n;
    assert_eq!(
        dequant.len(),
        n0 * n0,
        "dequant のサイズが n0 x n0 と一致しない"
    );
    if lossless {
        assert_eq!(n, 2, "ロスレス変換（WHT）は 4x4 のみサポートする");
    }

    let mut t = vec![0i64; n0];

    // 行変換（row transform）: i = 0..(n0-1)。
    for i in 0..n0 {
        t[..n0].copy_from_slice(&dequant[i * n0..(i + 1) * n0]);
        if lossless {
            let arr: &mut [i64; 4] = (&mut t[..4]).try_into().unwrap();
            iwht4(arr, 2);
        } else {
            match tx_type {
                TxType::DctDct | TxType::AdstDct => {
                    idct_permute(&mut t, n);
                    idct(&mut t, n);
                }
                TxType::DctAdst | TxType::AdstAdst => {
                    iadst(&mut t, n);
                }
            }
        }
        dequant[i * n0..(i + 1) * n0].copy_from_slice(&t[..n0]);
    }

    // 列変換（column transform）: j = 0..(n0-1)。
    let shift = (n + 2).min(6);
    for j in 0..n0 {
        for i in 0..n0 {
            t[i] = dequant[i * n0 + j];
        }
        if lossless {
            let arr: &mut [i64; 4] = (&mut t[..4]).try_into().unwrap();
            iwht4(arr, 0);
        } else {
            match tx_type {
                TxType::DctDct | TxType::DctAdst => {
                    idct_permute(&mut t, n);
                    idct(&mut t, n);
                }
                TxType::AdstDct | TxType::AdstAdst => {
                    iadst(&mut t, n);
                }
            }
        }
        if lossless {
            for i in 0..n0 {
                dequant[i * n0 + j] = t[i];
            }
        } else {
            for i in 0..n0 {
                dequant[i * n0 + j] = round2(t[i], shift);
            }
        }
    }
}

#[cfg(test)]
mod tests;
