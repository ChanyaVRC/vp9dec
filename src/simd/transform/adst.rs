//! 8-bit AVX2 inverse ADST networks (`iadst4/8/16`, spec §8.7.1.4-§8.7.1.9) on 8-lane `i32`
//! element vectors.
//!
//! Overflow safety at `bit_depth == 8` (from spec §8.7.1.1 / §8.7.2 conformance: every stored `T`
//! value fits signed `8 + BitDepth == 16` bits, so `|T| <= 2^15`; `|cos64| <= 2^14`; max
//! `|cos64| + |sin64| == 23170`):
//! - every `SB` product `T * cos64` is `<= 2^29`, so `_mm256_mullo_epi32` is exact;
//! - every unrounded `S` value is `<= 2^15 * 23170 < 2^30`, inside the `i32` lanes;
//! - every `SH` sum `S[a] +/- S[b]` (+ the `2^13` round) is `< 2^31`;
//! - the `iadst4` chains peak at `(SINPI_1 + SINPI_2 + SINPI_3 + SINPI_4) * 2^15 < 2^31`, and
//!   `SINPI_3_9 * (T0 - T2 + T3) <= 13377 * 3 * 2^15 < 2^31`.
//!
//! At 10/12-bit, the spec's bound on `S` is `24 + BitDepth` (up to 36) bits -- beyond `i32` lane
//! storage -- so `tile/residual.rs` routes 10/12-bit ADST to `adst_hbd.rs` (i64 lanes).

use std::arch::x86_64::*;

use super::dct::{b_op_simd, h_op_simd, round2_14};

/// Vector `SB(a, b, angle, flip)` (`transform.rs::sb_op`, spec §8.7.1.1) on 8 rows: the
/// butterfly rotation into the high-precision `S` array, unrounded. 8-bit only.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `a, b < t.len().min(s.len())`.
#[inline]
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn sb_op_simd(
    s: &mut [__m256i],
    t: &[__m256i],
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    let cos = _mm256_set1_epi32(crate::transform::cos64(angle) as i32);
    let sin = _mm256_set1_epi32(crate::transform::sin64(angle) as i32);
    let ta = t[a];
    let tb = t[b];
    let sa = _mm256_sub_epi32(_mm256_mullo_epi32(ta, cos), _mm256_mullo_epi32(tb, sin));
    let sb = _mm256_add_epi32(_mm256_mullo_epi32(ta, sin), _mm256_mullo_epi32(tb, cos));
    if flip {
        s[a] = sb;
        s[b] = sa;
    } else {
        s[a] = sa;
        s[b] = sb;
    }
}

/// Vector `SH(a, b)` (`transform.rs::sh_op`, spec §8.7.1.1) on 8 rows: Hadamard + `round2(14)`
/// from `S` back into `T`. 8-bit only.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `a, b < t.len().min(s.len())`.
#[inline]
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn sh_op_simd(t: &mut [__m256i], s: &[__m256i], a: usize, b: usize) {
    t[a] = round2_14(_mm256_add_epi32(s[a], s[b]));
    t[b] = round2_14(_mm256_sub_epi32(s[a], s[b]));
}

/// Vector `adst_input_permute` (`transform.rs`, spec §8.7.1.4): pure reorder, no arithmetic
/// (lane-agnostic; shared with `adst_hbd`).
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `n` is 3 or 4, and `t.len() >= (1 << n)`.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn adst_input_permute_simd(t: &mut [__m256i], n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let mut copy_t = [_mm256_setzero_si256(); 16];
    copy_t[..n0].copy_from_slice(&t[..n0]);
    for i in 0..n1 {
        t[2 * i] = copy_t[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy_t[2 * i];
    }
}

/// Vector `adst_output_permute` (`transform.rs`, spec §8.7.1.5): pure reorder, no arithmetic
/// (`n` is 3 or 4; lane-agnostic, shared with `adst_hbd`).
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `n` is 3 or 4, and `t.len() >= (1 << n)`.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn adst_output_permute_simd(t: &mut [__m256i], n: u32) {
    if n == 4 {
        let mut copy_t = [_mm256_setzero_si256(); 16];
        copy_t.copy_from_slice(&t[..16]);
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
        let mut copy_t = [_mm256_setzero_si256(); 8];
        copy_t.copy_from_slice(&t[..8]);
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

/// Vector inverse ADST4 (`transform.rs::iadst4_impl`, spec §8.7.1.6) on 8 rows.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `t.len() >= 4`.
#[target_feature(enable = "avx2")]
unsafe fn iadst4_simd(t: &mut [__m256i]) {
    use crate::transform::{SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9};
    let mul = |v: __m256i, c: i64| _mm256_mullo_epi32(v, _mm256_set1_epi32(c as i32));
    let s0 = mul(t[0], SINPI_1_9);
    let s1 = mul(t[0], SINPI_2_9);
    let s2 = mul(t[1], SINPI_3_9);
    let s3 = mul(t[2], SINPI_4_9);
    let s4 = mul(t[2], SINPI_1_9);
    let s5 = mul(t[3], SINPI_2_9);
    let s6 = mul(t[3], SINPI_4_9);
    let v = _mm256_add_epi32(_mm256_sub_epi32(t[0], t[2]), t[3]);
    let s7 = mul(v, SINPI_3_9);
    let x0 = _mm256_add_epi32(_mm256_add_epi32(s0, s3), s5);
    let x1 = _mm256_sub_epi32(_mm256_sub_epi32(s1, s4), s6);
    let x2 = s7;
    let x3 = s2;
    let o0 = _mm256_add_epi32(x0, x3);
    let o1 = _mm256_add_epi32(x1, x3);
    let o2 = x2;
    let o3 = _mm256_sub_epi32(_mm256_add_epi32(x0, x1), x3);
    t[0] = round2_14(o0);
    t[1] = round2_14(o1);
    t[2] = round2_14(o2);
    t[3] = round2_14(o3);
}

/// Vector inverse ADST8 (`transform.rs::iadst8_impl`, spec §8.7.1.7) on 8 rows.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `t.len() >= 8`.
#[target_feature(enable = "avx2")]
unsafe fn iadst8_simd(t: &mut [__m256i]) {
    adst_input_permute_simd(t, 3);
    let mut s = [_mm256_setzero_si256(); 8];

    for i in 0..4usize {
        sb_op_simd(&mut s, t, 2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
    }
    for i in 0..4usize {
        sh_op_simd(t, &s, i, 4 + i);
    }
    for i in 0..2usize {
        sb_op_simd(&mut s, t, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
    }
    for i in 0..2usize {
        sh_op_simd(t, &s, 4 + i, 6 + i);
    }
    for i in 0..2usize {
        h_op_simd(t, i, 2 + i, false);
    }
    for i in 0..2usize {
        b_op_simd(t, 2 + 4 * i, 3 + 4 * i, 16, true);
    }

    adst_output_permute_simd(t, 3);

    let zero = _mm256_setzero_si256();
    for i in 0..4usize {
        t[1 + 2 * i] = _mm256_sub_epi32(zero, t[1 + 2 * i]);
    }
}

/// Vector inverse ADST16 (`transform.rs::iadst16_impl`, spec §8.7.1.8) on 8 rows.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `t.len() >= 16`.
#[target_feature(enable = "avx2")]
unsafe fn iadst16_simd(t: &mut [__m256i]) {
    adst_input_permute_simd(t, 4);
    let mut s = [_mm256_setzero_si256(); 16];

    for i in 0..8usize {
        sb_op_simd(&mut s, t, 2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
    }
    for i in 0..8usize {
        sh_op_simd(t, &s, i, 8 + i);
    }
    for i in 0..4usize {
        sb_op_simd(&mut s, t, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    for i in 0..4usize {
        sh_op_simd(t, &s, 8 + i, 12 + i);
    }
    for i in 0..4usize {
        h_op_simd(t, i, 4 + i, false);
    }
    for i in 0..2usize {
        for j in 0..2usize {
            sb_op_simd(
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
            sh_op_simd(t, &s, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            h_op_simd(t, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            let angle = 48 + 64 * (i ^ j) as i32;
            b_op_simd(t, 2 + 4 * j + 8 * i, 3 + 4 * j + 8 * i, angle, false);
        }
    }

    adst_output_permute_simd(t, 4);

    let zero = _mm256_setzero_si256();
    for i in 0..2usize {
        for j in 0..2usize {
            t[1 + 12 * j + 2 * i] = _mm256_sub_epi32(zero, t[1 + 12 * j + 2 * i]);
        }
    }
}

/// Vector `iadst(t, n)` (`transform.rs::iadst`, spec §8.7.1.9): ADST4/8/16 by size.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `2 <= n <= 4`, and `t.len() >= (1 << n)`.
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn iadst_simd(t: &mut [__m256i], n: u32) {
    match n {
        2 => iadst4_simd(t),
        3 => iadst8_simd(t),
        4 => iadst16_simd(t),
        _ => unreachable!("iadst_simd only supports n = 2..=4"),
    }
}
