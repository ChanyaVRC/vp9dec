//! 8-lane AVX2 inverse DCT butterfly network (spec §8.7.1.3), supporting both 8-bit (`HBD = false`,
//! `_mm256_mullo_epi32` products) and 10/12-bit (`HBD = true`, widened 32x32->64-bit products via
//! `_mm256_mul_epi32` + i64 rounding). Lane storage is `i32` at every bit depth per spec §8.7.1.1's
//! `8 + BitDepth <= 20`-bit bound on stored `T` values.

use std::arch::x86_64::*;

/// `round2(x, 14)` (`transform.rs::round2`) lane-wise; the 8-bit `b_op` and `sh_op` sums fit i32,
/// so `(x + 2^13) >> 14` arithmetic-shifted matches the scalar i64 `round2`.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn round2_14(x: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_add_epi32(x, _mm256_set1_epi32(1 << 13)), 14)
}

/// Vector `B(a, b, angle, flip)` butterfly (`transform.rs::b_op`) on 8 rows at once (8-bit).
#[inline]
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn b_op_simd(
    t: &mut [__m256i],
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    let cos = _mm256_set1_epi32(crate::transform::cos64(angle) as i32);
    let sin = _mm256_set1_epi32(crate::transform::sin64(angle) as i32);
    let ta = t[a];
    let tb = t[b];
    // x = ta*cos - tb*sin ; y = ta*sin + tb*cos ; each product fits i32 (mullo is exact there).
    let x = _mm256_sub_epi32(_mm256_mullo_epi32(ta, cos), _mm256_mullo_epi32(tb, sin));
    let y = _mm256_add_epi32(_mm256_mullo_epi32(ta, sin), _mm256_mullo_epi32(tb, cos));
    let na = round2_14(x);
    let nb = round2_14(y);
    if flip {
        t[a] = nb;
        t[b] = na;
    } else {
        t[a] = na;
        t[b] = nb;
    }
}

/// High-bit-depth `B(a, b, angle, flip)`: identical network arithmetic to [`b_op_simd`], but
/// the `t * cos64` products are computed with 32x32 -> 64-bit widening multiplies
/// (`_mm256_mul_epi32`) and rounded in i64, because at 10/12-bit a stored T value spans up to
/// signed `8 + BitDepth <= 20` bits (spec §8.7.1.1) and `t * cos64` (20b * 15b) would wrap
/// `_mm256_mullo_epi32`. Overflow-safe by that same bound: |t| < 2^19 and |cos64| <= 2^14, so
/// |ta*cos - tb*sin| + 2^13 < 2^35 fits i64 exactly. The `>> 14` uses a LOGICAL 64-bit shift
/// (AVX2 has no 64-bit arithmetic shift): logical and arithmetic shifts differ only in the top
/// 14 bits of the 64-bit result, and only the low 32 bits are kept -- exact whenever the true
/// `round2` result fits i32, which §8.7.1.1 guarantees (it is a stored T value).
#[inline]
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn b_op_simd_hbd(
    t: &mut [__m256i],
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    let cos = _mm256_set1_epi32(crate::transform::cos64(angle) as i32);
    let sin = _mm256_set1_epi32(crate::transform::sin64(angle) as i32);
    let ta = t[a];
    let tb = t[b];
    // `_mm256_mul_epi32` multiplies the (sign-extended) low 32 bits of each 64-bit lane, i.e.
    // i32 lanes 0/2/4/6. The logical 64-bit shift moves lanes 1/3/5/7 into those positions
    // (the zeroed high halves are ignored by the multiply).
    let ta_odd = _mm256_srli_epi64(ta, 32);
    let tb_odd = _mm256_srli_epi64(tb, 32);
    let round = _mm256_set1_epi64x(1 << 13);
    // x = ta*cos - tb*sin ; y = ta*sin + tb*cos, each + 2^13, in exact i64 (per even/odd half).
    let x_even = _mm256_add_epi64(
        _mm256_sub_epi64(_mm256_mul_epi32(ta, cos), _mm256_mul_epi32(tb, sin)),
        round,
    );
    let x_odd = _mm256_add_epi64(
        _mm256_sub_epi64(_mm256_mul_epi32(ta_odd, cos), _mm256_mul_epi32(tb_odd, sin)),
        round,
    );
    let y_even = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(ta, sin), _mm256_mul_epi32(tb, cos)),
        round,
    );
    let y_odd = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(ta_odd, sin), _mm256_mul_epi32(tb_odd, cos)),
        round,
    );
    // `round2(., 14)` per 64-bit lane (logical shift, low 32 bits exact -- see above), then
    // re-interleave the even/odd halves back into 8 i32 lanes.
    let combine = |even: __m256i, odd: __m256i| {
        _mm256_blend_epi32(
            _mm256_srli_epi64(even, 14),
            _mm256_slli_epi64(_mm256_srli_epi64(odd, 14), 32),
            0b1010_1010,
        )
    };
    let na = combine(x_even, x_odd);
    let nb = combine(y_even, y_odd);
    if flip {
        t[a] = nb;
        t[b] = na;
    } else {
        t[a] = na;
        t[b] = nb;
    }
}

/// Selects the depth-correct butterfly: `b_op_simd` (i32 products, bit_depth == 8) or
/// `b_op_simd_hbd` (widened i64 products, 10/12-bit). Monomorphized away -- the 8-bit network
/// is unchanged.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn b_op_simd_depth<const HBD: bool>(
    t: &mut [__m256i],
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    if HBD {
        b_op_simd_hbd(t, a, b, angle, flip);
    } else {
        b_op_simd(t, a, b, angle, flip);
    }
}

/// Vector `H(a, b, flip)` Hadamard rotation (`transform.rs::h_op`) on 8 rows.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn h_op_simd(t: &mut [__m256i], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = _mm256_add_epi32(x, y);
    t[b] = _mm256_sub_epi32(x, y);
}

/// Vector `idct_permute` (`transform.rs::idct_permute`): bit-reversal reorder of the element
/// vectors (no arithmetic, so lane-agnostic and shared between i32 and i64 networks).
#[inline]
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn idct_permute_simd(t: &mut [__m256i], n: u32) {
    let n0 = 1usize << n;
    let mut copy_t = [_mm256_setzero_si256(); 32];
    copy_t[..n0].copy_from_slice(&t[..n0]);
    for (i, dst) in t[..n0].iter_mut().enumerate() {
        *dst = copy_t[crate::transform::brev(n, i)];
    }
}

/// Vector inverse-DCT butterfly network -- a verbatim mirror of `transform.rs::idct` (spec
/// §8.7.1.3), operating on 8-row i32 element vectors. Assumes `idct_permute_simd` ran first.
/// `HBD` selects the butterfly's multiply width (see [`b_op_simd_depth`]); the network
/// structure is identical either way.
#[target_feature(enable = "avx2")]
pub(in crate::simd) unsafe fn idct_simd<const HBD: bool>(t: &mut [__m256i], n: u32) {
    let n0 = 1i64 << n;
    let n1 = 1i64 << (n - 1);
    let n2 = 1i64 << (n - 2);

    if n == 2 {
        b_op_simd_depth::<HBD>(t, 0, 1, 16, true);
    } else {
        idct_simd::<HBD>(t, n - 1);
    }

    for i in 0..n2 {
        let a = (n1 + i) as usize;
        let b = (n0 - 1 - i) as usize;
        let angle = 32 - crate::transform::brev(5, a) as i32;
        b_op_simd_depth::<HBD>(t, a, b, angle, false);
    }

    if n >= 3 {
        let n3 = 1i64 << (n - 3);
        for i in 0..n3 {
            for j in 0..2i64 {
                let a = (n1 + 4 * i + 2 * j) as usize;
                let b = (n1 + 1 + 4 * i + 2 * j) as usize;
                h_op_simd(t, a, b, j == 1);
            }
        }
    }

    if n == 5 {
        for i in 0..2i64 {
            for j in 0..2i64 {
                let a = (n0 - n as i64 + 3 - n2 * j - 4 * i) as usize;
                let b = (n1 + n as i64 - 4 + n2 * j + 4 * i) as usize;
                let angle = 28 - 16 * i as i32 + 56 * j as i32;
                b_op_simd_depth::<HBD>(t, a, b, angle, true);
            }
        }
        let n3 = 1i64 << (n - 3);
        for i in 0..2i64 {
            for j in 0..4i64 {
                let a = (n1 + n3 * j + i) as usize;
                let b = (n1 + n2 - 5 + n3 * j - i) as usize;
                h_op_simd(t, a, b, (j & 1) == 1);
            }
        }
    }

    if n >= 4 {
        let imax_a: i64 = if n == 5 { 1 } else { 0 };
        for i in 0..=imax_a {
            for j in 0..2i64 {
                let a = (n0 - n as i64 + 2 - i - n2 * j) as usize;
                let b = (n1 + n as i64 - 3 + i + n2 * j) as usize;
                let angle = 24 + 48 * j as i32;
                b_op_simd_depth::<HBD>(t, a, b, angle, true);
            }
        }
        let imax_b: i64 = 2 * n as i64 - 7;
        for j in 0..2i64 {
            for i in 0..=imax_b {
                let a = (n1 + n2 * j + i) as usize;
                let b = (n1 + n2 - 1 + n2 * j - i) as usize;
                h_op_simd(t, a, b, (j & 1) == 1);
            }
        }
    }

    if n >= 3 {
        let n3 = 1i64 << (n - 3);
        for i in 0..n3 {
            let a = (n0 - n3 - 1 - i) as usize;
            let b = (n1 + n3 + i) as usize;
            b_op_simd_depth::<HBD>(t, a, b, 16, true);
        }
    }

    for i in 0..n1 {
        let a = i as usize;
        let b = (n0 - 1 - i) as usize;
        h_op_simd(t, a, b, false);
    }
}
