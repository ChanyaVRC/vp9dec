//! AVX2 SIMD mirror of `predict::block_inter_predict`'s scalar two-pass 8-tap subpel
//! convolution (spec §8.5.2.4): specialized kernels for the common UNSCALED case
//! (`x_step == y_step == 16`, i.e. reference frame same size as the current frame) plus a
//! general edge-clamping kernel ([`block_inter_predict_scaled_u16_avx2`]). The direct-to-plane
//! `*_u16_avx2` entries write `u16` samples straight into the destination plane (with optional
//! compound averaging via `VPAVGW` / `PAVGW`, matching `round2(a + b, 1)`), dispatch full-pel
//! copy and 1D horizontal/vertical fast paths when `fx == 0` or `fy == 0`, use `_mm256_madd_epi16`
//! for 2 taps per instruction across all bit depths (8/10/12-bit), and use uninitialized stack
//! scratch so small blocks never pay a 34 KB zero-fill. `predict.rs` owns the dispatch and the
//! scalar fallback; this module owns only the vector kernels.

use crate::prob_tables::BILINEAR;
use crate::subpel::{MAX_BLOCK_DIM, MAX_INTERMEDIATE_HEIGHT, SUBPEL_FILTERS};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

/// Packs two `i32` subpel filter coefficients (each in `-24..=128`) as repeated `(a, b)` `i16`
/// pairs in a 256-bit register for `_mm256_madd_epi16`.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pair_coeff_256(a: i32, b: i32) -> __m256i {
    let lo = a as i16;
    let hi = b as i16;
    _mm256_setr_epi16(
        lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi,
    )
}

/// 128-bit counterpart of [`pair_coeff_256`] for `_mm_madd_epi16`.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pair_coeff_128(a: i32, b: i32) -> __m128i {
    let lo = a as i16;
    let hi = b as i16;
    _mm_setr_epi16(lo, hi, lo, hi, lo, hi, lo, hi)
}

/// Rounds (`(x + 64) >> 7`) and clips two 8-lane `i32` accumulators (`acc_lo` holding lanes
/// `0..4, 8..12` and `acc_hi` holding lanes `4..8, 12..16`) and packs them into 16 contiguous
/// `u16` lanes in exact order `0..16` (in-lane `VPACKUSDW` needs no cross-lane permute).
///
/// # Safety
/// Caller must confirm `avx2_enabled()`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round_clip_pack_16(
    acc_lo: __m256i,
    acc_hi: __m256i,
    round_add: __m256i,
    zero: __m256i,
    max_clip: __m256i,
) -> __m256i {
    let r_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, round_add), 7);
    let r_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, round_add), 7);
    let c_lo = _mm256_min_epi32(_mm256_max_epi32(r_lo, zero), max_clip);
    let c_hi = _mm256_min_epi32(_mm256_max_epi32(r_hi, zero), max_clip);
    _mm256_packus_epi32(c_lo, c_hi)
}

/// Rounds (`(x + 64) >> 7`), clips, and packs two 4-lane `i32` accumulators into 8 contiguous
/// `u16` lanes.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round_clip_pack_8(
    acc_lo: __m128i,
    acc_hi: __m128i,
    round_add: __m128i,
    zero: __m128i,
    max_clip: __m128i,
) -> __m128i {
    let r_lo = _mm_srai_epi32(_mm_add_epi32(acc_lo, round_add), 7);
    let r_hi = _mm_srai_epi32(_mm_add_epi32(acc_hi, round_add), 7);
    let c_lo = _mm_min_epi32(_mm_max_epi32(r_lo, zero), max_clip);
    let c_hi = _mm_min_epi32(_mm_max_epi32(r_hi, zero), max_clip);
    _mm_packus_epi32(c_lo, c_hi)
}

/// Stores 16 `u16` values into `dst`, or averages them with existing `dst` via `VPAVGW`
/// (`(dst + v + 1) >> 1 == round2(dst + v, 1)`) when `COMPOUND` is true.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `dst` valid for 16 `u16` writes (and initialized
/// reads when `COMPOUND`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_or_avg_16<const COMPOUND: bool>(dst: *mut u16, v: __m256i) {
    let out = if COMPOUND {
        let cur = _mm256_loadu_si256(dst as *const __m256i);
        _mm256_avg_epu16(cur, v)
    } else {
        v
    };
    _mm256_storeu_si256(dst as *mut __m256i, out);
}

/// 8-sample counterpart of [`store_or_avg_16`].
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `dst` valid for 8 `u16` writes (and initialized
/// reads when `COMPOUND`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_or_avg_8<const COMPOUND: bool>(dst: *mut u16, v: __m128i) {
    let out = if COMPOUND {
        let cur = _mm_loadu_si128(dst as *const __m128i);
        _mm_avg_epu16(cur, v)
    } else {
        v
    };
    _mm_storeu_si128(dst as *mut __m128i, out);
}

/// 4-sample counterpart of [`store_or_avg_16`].
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and `dst` valid for 4 `u16` writes (and initialized
/// reads when `COMPOUND`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_or_avg_4<const COMPOUND: bool>(dst: *mut u16, v: __m128i) {
    let out = if COMPOUND {
        let cur = _mm_loadl_epi64(dst as *const __m128i);
        _mm_avg_epu16(cur, v)
    } else {
        v
    };
    _mm_storel_epi64(dst as *mut __m128i, out);
}

/// Horizontal 1D subpel filter over `rows` rows of `src` (stride `src_stride`), writing `w`
/// (`4` or multiple of `8`) `u16` samples per row into `dst` (stride `dst_stride`), with optional
/// compound averaging. `src` must point to `(row0, col0)` where `col0` is tap 0 (`(x >> 4) - 3`).
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `w` is 4 or a multiple of 8, `src` valid for `rows`
/// rows of `w + 7` `u16` reads at `src_stride`, and `dst` valid for `rows` rows of `w` `u16`
/// writes at `dst_stride` (initialized when `COMPOUND`).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn convolve_horiz_u16_avx2<const COMPOUND: bool>(
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    dst_stride: usize,
    w: usize,
    rows: usize,
    coeffs: &[i32; 8],
    is_bilinear: bool,
    max_val: i32,
) {
    if w == 4 {
        let round_add = _mm_set1_epi32(64);
        let zero = _mm_setzero_si128();
        let max_clip = _mm_set1_epi32(max_val);
        if is_bilinear {
            let c34 = pair_coeff_128(coeffs[3], coeffs[4]);
            for r in 0..rows {
                let sp = src.add(r * src_stride);
                let s3 = _mm_loadl_epi64(sp.add(3) as *const __m128i);
                let s4 = _mm_loadl_epi64(sp.add(4) as *const __m128i);
                let acc = _mm_madd_epi16(_mm_unpacklo_epi16(s3, s4), c34);
                let res = _mm_srai_epi32(_mm_add_epi32(acc, round_add), 7);
                let clipped = _mm_min_epi32(_mm_max_epi32(res, zero), max_clip);
                store_or_avg_4::<COMPOUND>(
                    dst.add(r * dst_stride),
                    _mm_packus_epi32(clipped, clipped),
                );
            }
        } else {
            let c01 = pair_coeff_128(coeffs[0], coeffs[1]);
            let c23 = pair_coeff_128(coeffs[2], coeffs[3]);
            let c45 = pair_coeff_128(coeffs[4], coeffs[5]);
            let c67 = pair_coeff_128(coeffs[6], coeffs[7]);
            for r in 0..rows {
                let sp = src.add(r * src_stride);
                let s0 = _mm_loadl_epi64(sp as *const __m128i);
                let s1 = _mm_loadl_epi64(sp.add(1) as *const __m128i);
                let s2 = _mm_loadl_epi64(sp.add(2) as *const __m128i);
                let s3 = _mm_loadl_epi64(sp.add(3) as *const __m128i);
                let s4 = _mm_loadl_epi64(sp.add(4) as *const __m128i);
                let s5 = _mm_loadl_epi64(sp.add(5) as *const __m128i);
                let s6 = _mm_loadl_epi64(sp.add(6) as *const __m128i);
                let s7 = _mm_loadl_epi64(sp.add(7) as *const __m128i);
                let mut acc = _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), c01);
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), c23));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), c45));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), c67));
                let res = _mm_srai_epi32(_mm_add_epi32(acc, round_add), 7);
                let clipped = _mm_min_epi32(_mm_max_epi32(res, zero), max_clip);
                store_or_avg_4::<COMPOUND>(
                    dst.add(r * dst_stride),
                    _mm_packus_epi32(clipped, clipped),
                );
            }
        }
        return;
    }

    if w == 8 {
        let round_add = _mm_set1_epi32(64);
        let zero = _mm_setzero_si128();
        let max_clip = _mm_set1_epi32(max_val);
        if is_bilinear {
            let c34 = pair_coeff_128(coeffs[3], coeffs[4]);
            for r in 0..rows {
                let sp = src.add(r * src_stride);
                let s3 = _mm_loadu_si128(sp.add(3) as *const __m128i);
                let s4 = _mm_loadu_si128(sp.add(4) as *const __m128i);
                let acc_lo = _mm_madd_epi16(_mm_unpacklo_epi16(s3, s4), c34);
                let acc_hi = _mm_madd_epi16(_mm_unpackhi_epi16(s3, s4), c34);
                let packed = round_clip_pack_8(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_8::<COMPOUND>(dst.add(r * dst_stride), packed);
            }
        } else {
            let c01 = pair_coeff_128(coeffs[0], coeffs[1]);
            let c23 = pair_coeff_128(coeffs[2], coeffs[3]);
            let c45 = pair_coeff_128(coeffs[4], coeffs[5]);
            let c67 = pair_coeff_128(coeffs[6], coeffs[7]);
            for r in 0..rows {
                let sp = src.add(r * src_stride);
                let s0 = _mm_loadu_si128(sp as *const __m128i);
                let s1 = _mm_loadu_si128(sp.add(1) as *const __m128i);
                let s2 = _mm_loadu_si128(sp.add(2) as *const __m128i);
                let s3 = _mm_loadu_si128(sp.add(3) as *const __m128i);
                let s4 = _mm_loadu_si128(sp.add(4) as *const __m128i);
                let s5 = _mm_loadu_si128(sp.add(5) as *const __m128i);
                let s6 = _mm_loadu_si128(sp.add(6) as *const __m128i);
                let s7 = _mm_loadu_si128(sp.add(7) as *const __m128i);
                let mut acc_lo = _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), c01);
                let mut acc_hi = _mm_madd_epi16(_mm_unpackhi_epi16(s0, s1), c01);
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), c23));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(s2, s3), c23));
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), c45));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(s4, s5), c45));
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), c67));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(s6, s7), c67));
                let packed = round_clip_pack_8(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_8::<COMPOUND>(dst.add(r * dst_stride), packed);
            }
        }
        return;
    }

    // w is 16, 32, or 64 (multiple of 16): process 16 pixels per iteration with 256-bit AVX2.
    debug_assert_eq!(w % 16, 0);
    let round_add = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let max_clip = _mm256_set1_epi32(max_val);
    if is_bilinear {
        let c34 = pair_coeff_256(coeffs[3], coeffs[4]);
        for r in 0..rows {
            let row_src = src.add(r * src_stride);
            let row_dst = dst.add(r * dst_stride);
            let mut c = 0usize;
            while c < w {
                let sp = row_src.add(c);
                let s3 = _mm256_loadu_si256(sp.add(3) as *const __m256i);
                let s4 = _mm256_loadu_si256(sp.add(4) as *const __m256i);
                let acc_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(s3, s4), c34);
                let acc_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(s3, s4), c34);
                let packed = round_clip_pack_16(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_16::<COMPOUND>(row_dst.add(c), packed);
                c += 16;
            }
        }
    } else {
        let c01 = pair_coeff_256(coeffs[0], coeffs[1]);
        let c23 = pair_coeff_256(coeffs[2], coeffs[3]);
        let c45 = pair_coeff_256(coeffs[4], coeffs[5]);
        let c67 = pair_coeff_256(coeffs[6], coeffs[7]);
        for r in 0..rows {
            let row_src = src.add(r * src_stride);
            let row_dst = dst.add(r * dst_stride);
            let mut c = 0usize;
            while c < w {
                let sp = row_src.add(c);
                let s0 = _mm256_loadu_si256(sp as *const __m256i);
                let s1 = _mm256_loadu_si256(sp.add(1) as *const __m256i);
                let s2 = _mm256_loadu_si256(sp.add(2) as *const __m256i);
                let s3 = _mm256_loadu_si256(sp.add(3) as *const __m256i);
                let s4 = _mm256_loadu_si256(sp.add(4) as *const __m256i);
                let s5 = _mm256_loadu_si256(sp.add(5) as *const __m256i);
                let s6 = _mm256_loadu_si256(sp.add(6) as *const __m256i);
                let s7 = _mm256_loadu_si256(sp.add(7) as *const __m256i);
                let mut acc_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(s0, s1), c01);
                let mut acc_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(s0, s1), c01);
                acc_lo = _mm256_add_epi32(
                    acc_lo,
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(s2, s3), c23),
                );
                acc_hi = _mm256_add_epi32(
                    acc_hi,
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(s2, s3), c23),
                );
                acc_lo = _mm256_add_epi32(
                    acc_lo,
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(s4, s5), c45),
                );
                acc_hi = _mm256_add_epi32(
                    acc_hi,
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(s4, s5), c45),
                );
                acc_lo = _mm256_add_epi32(
                    acc_lo,
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(s6, s7), c67),
                );
                acc_hi = _mm256_add_epi32(
                    acc_hi,
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(s6, s7), c67),
                );
                let packed = round_clip_pack_16(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_16::<COMPOUND>(row_dst.add(c), packed);
                c += 16;
            }
        }
    }
}

/// Vertical 1D subpel filter over `h` output rows reading from `src` (stride `src_stride`,
/// where row `0` corresponds to tap `0` of output row `0`, i.e. `y0 - 3`), writing `w` (`4` or
/// multiple of `8`) `u16` samples per row into `dst` (stride `dst_stride`), with optional compound
/// averaging.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `w` is 4 or a multiple of 8, `src` valid for `h + 7`
/// rows of `w` `u16` reads at `src_stride`, and `dst` valid for `h` rows of `w` `u16` writes at
/// `dst_stride` (initialized when `COMPOUND`).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn convolve_vert_u16_avx2<const COMPOUND: bool>(
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    dst_stride: usize,
    w: usize,
    h: usize,
    coeffs: &[i32; 8],
    is_bilinear: bool,
    max_val: i32,
) {
    if w == 4 {
        let round_add = _mm_set1_epi32(64);
        let zero = _mm_setzero_si128();
        let max_clip = _mm_set1_epi32(max_val);
        if is_bilinear {
            let c34 = pair_coeff_128(coeffs[3], coeffs[4]);
            for r in 0..h {
                let s3 = _mm_loadl_epi64(src.add((r + 3) * src_stride) as *const __m128i);
                let s4 = _mm_loadl_epi64(src.add((r + 4) * src_stride) as *const __m128i);
                let acc = _mm_madd_epi16(_mm_unpacklo_epi16(s3, s4), c34);
                let res = _mm_srai_epi32(_mm_add_epi32(acc, round_add), 7);
                let clipped = _mm_min_epi32(_mm_max_epi32(res, zero), max_clip);
                store_or_avg_4::<COMPOUND>(
                    dst.add(r * dst_stride),
                    _mm_packus_epi32(clipped, clipped),
                );
            }
        } else {
            let c01 = pair_coeff_128(coeffs[0], coeffs[1]);
            let c23 = pair_coeff_128(coeffs[2], coeffs[3]);
            let c45 = pair_coeff_128(coeffs[4], coeffs[5]);
            let c67 = pair_coeff_128(coeffs[6], coeffs[7]);
            for r in 0..h {
                let s0 = _mm_loadl_epi64(src.add(r * src_stride) as *const __m128i);
                let s1 = _mm_loadl_epi64(src.add((r + 1) * src_stride) as *const __m128i);
                let s2 = _mm_loadl_epi64(src.add((r + 2) * src_stride) as *const __m128i);
                let s3 = _mm_loadl_epi64(src.add((r + 3) * src_stride) as *const __m128i);
                let s4 = _mm_loadl_epi64(src.add((r + 4) * src_stride) as *const __m128i);
                let s5 = _mm_loadl_epi64(src.add((r + 5) * src_stride) as *const __m128i);
                let s6 = _mm_loadl_epi64(src.add((r + 6) * src_stride) as *const __m128i);
                let s7 = _mm_loadl_epi64(src.add((r + 7) * src_stride) as *const __m128i);
                let mut acc = _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), c01);
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), c23));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), c45));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), c67));
                let res = _mm_srai_epi32(_mm_add_epi32(acc, round_add), 7);
                let clipped = _mm_min_epi32(_mm_max_epi32(res, zero), max_clip);
                store_or_avg_4::<COMPOUND>(
                    dst.add(r * dst_stride),
                    _mm_packus_epi32(clipped, clipped),
                );
            }
        }
        return;
    }

    if w == 8 {
        let round_add = _mm_set1_epi32(64);
        let zero = _mm_setzero_si128();
        let max_clip = _mm_set1_epi32(max_val);
        if is_bilinear {
            let c34 = pair_coeff_128(coeffs[3], coeffs[4]);
            for r in 0..h {
                let s3 = _mm_loadu_si128(src.add((r + 3) * src_stride) as *const __m128i);
                let s4 = _mm_loadu_si128(src.add((r + 4) * src_stride) as *const __m128i);
                let acc_lo = _mm_madd_epi16(_mm_unpacklo_epi16(s3, s4), c34);
                let acc_hi = _mm_madd_epi16(_mm_unpackhi_epi16(s3, s4), c34);
                let packed = round_clip_pack_8(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_8::<COMPOUND>(dst.add(r * dst_stride), packed);
            }
        } else {
            let c01 = pair_coeff_128(coeffs[0], coeffs[1]);
            let c23 = pair_coeff_128(coeffs[2], coeffs[3]);
            let c45 = pair_coeff_128(coeffs[4], coeffs[5]);
            let c67 = pair_coeff_128(coeffs[6], coeffs[7]);
            for r in 0..h {
                let s0 = _mm_loadu_si128(src.add(r * src_stride) as *const __m128i);
                let s1 = _mm_loadu_si128(src.add((r + 1) * src_stride) as *const __m128i);
                let s2 = _mm_loadu_si128(src.add((r + 2) * src_stride) as *const __m128i);
                let s3 = _mm_loadu_si128(src.add((r + 3) * src_stride) as *const __m128i);
                let s4 = _mm_loadu_si128(src.add((r + 4) * src_stride) as *const __m128i);
                let s5 = _mm_loadu_si128(src.add((r + 5) * src_stride) as *const __m128i);
                let s6 = _mm_loadu_si128(src.add((r + 6) * src_stride) as *const __m128i);
                let s7 = _mm_loadu_si128(src.add((r + 7) * src_stride) as *const __m128i);
                let mut acc_lo = _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), c01);
                let mut acc_hi = _mm_madd_epi16(_mm_unpackhi_epi16(s0, s1), c01);
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), c23));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(s2, s3), c23));
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), c45));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(s4, s5), c45));
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), c67));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(s6, s7), c67));
                let packed = round_clip_pack_8(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_8::<COMPOUND>(dst.add(r * dst_stride), packed);
            }
        }
        return;
    }

    // w is 16, 32, or 64.
    debug_assert_eq!(w % 16, 0);
    let round_add = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let max_clip = _mm256_set1_epi32(max_val);
    if is_bilinear {
        let c34 = pair_coeff_256(coeffs[3], coeffs[4]);
        for r in 0..h {
            let p3 = src.add((r + 3) * src_stride);
            let p4 = src.add((r + 4) * src_stride);
            let row_dst = dst.add(r * dst_stride);
            let mut c = 0usize;
            while c < w {
                let s3 = _mm256_loadu_si256(p3.add(c) as *const __m256i);
                let s4 = _mm256_loadu_si256(p4.add(c) as *const __m256i);
                let acc_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(s3, s4), c34);
                let acc_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(s3, s4), c34);
                let packed = round_clip_pack_16(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_16::<COMPOUND>(row_dst.add(c), packed);
                c += 16;
            }
        }
    } else {
        let c01 = pair_coeff_256(coeffs[0], coeffs[1]);
        let c23 = pair_coeff_256(coeffs[2], coeffs[3]);
        let c45 = pair_coeff_256(coeffs[4], coeffs[5]);
        let c67 = pair_coeff_256(coeffs[6], coeffs[7]);
        for r in 0..h {
            let p0 = src.add(r * src_stride);
            let p1 = src.add((r + 1) * src_stride);
            let p2 = src.add((r + 2) * src_stride);
            let p3 = src.add((r + 3) * src_stride);
            let p4 = src.add((r + 4) * src_stride);
            let p5 = src.add((r + 5) * src_stride);
            let p6 = src.add((r + 6) * src_stride);
            let p7 = src.add((r + 7) * src_stride);
            let row_dst = dst.add(r * dst_stride);
            let mut c = 0usize;
            while c < w {
                let s0 = _mm256_loadu_si256(p0.add(c) as *const __m256i);
                let s1 = _mm256_loadu_si256(p1.add(c) as *const __m256i);
                let s2 = _mm256_loadu_si256(p2.add(c) as *const __m256i);
                let s3 = _mm256_loadu_si256(p3.add(c) as *const __m256i);
                let s4 = _mm256_loadu_si256(p4.add(c) as *const __m256i);
                let s5 = _mm256_loadu_si256(p5.add(c) as *const __m256i);
                let s6 = _mm256_loadu_si256(p6.add(c) as *const __m256i);
                let s7 = _mm256_loadu_si256(p7.add(c) as *const __m256i);
                let mut acc_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(s0, s1), c01);
                let mut acc_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(s0, s1), c01);
                acc_lo = _mm256_add_epi32(
                    acc_lo,
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(s2, s3), c23),
                );
                acc_hi = _mm256_add_epi32(
                    acc_hi,
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(s2, s3), c23),
                );
                acc_lo = _mm256_add_epi32(
                    acc_lo,
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(s4, s5), c45),
                );
                acc_hi = _mm256_add_epi32(
                    acc_hi,
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(s4, s5), c45),
                );
                acc_lo = _mm256_add_epi32(
                    acc_lo,
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(s6, s7), c67),
                );
                acc_hi = _mm256_add_epi32(
                    acc_hi,
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(s6, s7), c67),
                );
                let packed = round_clip_pack_16(acc_lo, acc_hi, round_add, zero, max_clip);
                store_or_avg_16::<COMPOUND>(row_dst.add(c), packed);
                c += 16;
            }
        }
    }
}

/// Full-pel copy (or compound average via `VPAVGW`/`PAVGW`) for `fx == 0 && fy == 0`.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `w` is 4 or a multiple of 8, `src` valid for `h` rows
/// of `w` `u16` reads at `src_stride`, and `dst` valid for `h` rows of `w` `u16` writes at
/// `dst_stride` (initialized when `COMPOUND`).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn copy_u16_avx2<const COMPOUND: bool>(
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    match w {
        4 => {
            for r in 0..h {
                let v = _mm_loadl_epi64(src.add(r * src_stride) as *const __m128i);
                store_or_avg_4::<COMPOUND>(dst.add(r * dst_stride), v);
            }
        }
        8 => {
            for r in 0..h {
                let v = _mm_loadu_si128(src.add(r * src_stride) as *const __m128i);
                store_or_avg_8::<COMPOUND>(dst.add(r * dst_stride), v);
            }
        }
        _ => {
            debug_assert_eq!(w % 16, 0);
            for r in 0..h {
                let sp = src.add(r * src_stride);
                let dp = dst.add(r * dst_stride);
                let mut c = 0usize;
                while c < w {
                    let v = _mm256_loadu_si256(sp.add(c) as *const __m256i);
                    store_or_avg_16::<COMPOUND>(dp.add(c), v);
                    c += 16;
                }
            }
        }
    }
}

/// Direct-to-plane unscaled AVX2 inter prediction (`x_step == y_step == 16`, `w` is 4 or a
/// multiple of 8). Writes `u16` output straight into `dst_data` at `(dst_x, dst_y)` (or averages
/// with existing `dst_data` via `VPAVGW`/`PAVGW` when `COMPOUND == true`), dispatching full-pel
/// copy (`fx == 0 && fy == 0`), 1D horizontal (`fy == 0`), 1D vertical (`fx == 0`), and 2D
/// separable paths with uninitialized `u16` intermediate scratch.
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, that `dst_data` holds rows `dst_y..dst_y + h` x cols
/// `dst_x..dst_x + w` (stride `dst_width`), and that `ref_data` (stride `ref_width`) covers every
/// source pixel read by the active `(fx, fy)` branch:
/// - rows `y0..y0 + h` if `fy == 0`, else `(y0 - 3)..(y0 + h + 4)`;
/// - cols `x0..x0 + w` if `fx == 0`, else `(x0 - 3)..(x0 + w + 4)`.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_inter_predict_u16_avx2<const COMPOUND: bool>(
    ref_data: &[u16],
    ref_width: usize,
    dst_data: &mut [u16],
    dst_width: usize,
    dst_x: usize,
    dst_y: usize,
    x0: i64,
    y0: i64,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    interp_filter: u8,
    max_val: i32,
) {
    debug_assert!(dst_x + w <= dst_width);
    debug_assert!(h == 0 || (dst_y + h - 1) * dst_width + dst_x + w <= dst_data.len());
    let dst = dst_data.as_mut_ptr().add(dst_y * dst_width + dst_x);
    block_inter_predict_u16_ptr_avx2::<COMPOUND>(
        ref_data,
        ref_width,
        dst,
        dst_width,
        x0,
        y0,
        fx,
        fy,
        w,
        h,
        interp_filter,
        max_val,
    );
}

/// Raw-pointer core of [`block_inter_predict_u16_avx2`]. Taking `dst: *mut u16` allows callers
/// writing into uninitialized `MaybeUninit<u16>` stack scratch (`COMPOUND == false`) to avoid
/// constructing an invalid `&mut [u16]` over uninitialized memory.
///
/// # Safety
/// Same contract as [`block_inter_predict_u16_avx2`] with `dst` pointing to `(dst_x, dst_y)`
/// (writable for `h` rows of `w` samples at stride `dst_width`, and already initialized when
/// `COMPOUND == true`).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn block_inter_predict_u16_ptr_avx2<const COMPOUND: bool>(
    ref_data: &[u16],
    ref_width: usize,
    dst: *mut u16,
    dst_width: usize,
    x0: i64,
    y0: i64,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    interp_filter: u8,
    max_val: i32,
) {
    debug_assert!(w == 4 || w.is_multiple_of(8));
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    let is_bilinear = interp_filter == BILINEAR;
    let filters = &SUBPEL_FILTERS[interp_filter as usize];

    match (fx == 0, fy == 0) {
        (true, true) => {
            let src = ref_data
                .as_ptr()
                .add((y0 as usize) * ref_width + (x0 as usize));
            copy_u16_avx2::<COMPOUND>(src, ref_width, dst, dst_width, w, h);
        }
        (false, true) => {
            let src = ref_data
                .as_ptr()
                .add((y0 as usize) * ref_width + ((x0 - 3) as usize));
            convolve_horiz_u16_avx2::<COMPOUND>(
                src,
                ref_width,
                dst,
                dst_width,
                w,
                h,
                &filters[fx],
                is_bilinear,
                max_val,
            );
        }
        (true, false) => {
            let src = ref_data
                .as_ptr()
                .add(((y0 - 3) as usize) * ref_width + (x0 as usize));
            convolve_vert_u16_avx2::<COMPOUND>(
                src,
                ref_width,
                dst,
                dst_width,
                w,
                h,
                &filters[fy],
                is_bilinear,
                max_val,
            );
        }
        (false, false) => {
            // Uninitialized stack scratch sized for unscaled 2D max (71 * 64 u16 = 9 KB):
            // only the first `(h + 7) * w` entries are written by the horizontal pass and read
            // by the vertical pass.
            let mut intermediate =
                [MaybeUninit::<u16>::uninit(); (MAX_BLOCK_DIM + 7) * MAX_BLOCK_DIM];
            let tmp_ptr = intermediate.as_mut_ptr() as *mut u16;
            let src = ref_data
                .as_ptr()
                .add(((y0 - 3) as usize) * ref_width + ((x0 - 3) as usize));
            convolve_horiz_u16_avx2::<false>(
                src,
                ref_width,
                tmp_ptr,
                w,
                w,
                h + 7,
                &filters[fx],
                is_bilinear,
                max_val,
            );
            convolve_vert_u16_avx2::<COMPOUND>(
                tmp_ptr,
                w,
                dst,
                dst_width,
                w,
                h,
                &filters[fy],
                is_bilinear,
                max_val,
            );
        }
    }
}

/// Direct-to-plane edge-clamping AVX2 inter prediction for unscaled border blocks
/// (`x_step == y_step == 16`) AND scaled references (`x_step`/`y_step` != 16, SVC / resize).
/// Writes `u16` output straight into `dst_data` at `(dst_x, dst_y)` (or averages with existing
/// `dst_data` when `COMPOUND == true`), without zero-initializing the intermediate scratch.
///
/// # Safety
/// The caller must have confirmed `avx2_enabled()`, and must guarantee:
/// `ref_stride >= ref_width`; `ref_data.len() >= ref_stride * ref_height`;
/// `intermediate_height <= MAX_INTERMEDIATE_HEIGHT`; horizontal source span
/// `<= MAX_INTERMEDIATE_HEIGHT`; and `dst_data` holds rows `dst_y..dst_y + h` x cols
/// `dst_x..dst_x + w` (stride `dst_width`).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_inter_predict_scaled_u16_avx2<const COMPOUND: bool>(
    ref_data: &[u16],
    ref_width: usize,
    ref_stride: usize,
    ref_height: usize,
    dst_data: &mut [u16],
    dst_width: usize,
    dst_x: usize,
    dst_y: usize,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    intermediate_height: usize,
    interp_filter: u8,
    max_val: i32,
) {
    debug_assert!(w == 4 || w.is_multiple_of(8));
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert!(ref_stride >= ref_width);
    debug_assert!(ref_data.len() >= ref_stride * ref_height);
    debug_assert!(x_step >= 0 && y_step >= 0);

    let dst = dst_data.as_mut_ptr().add(dst_y * dst_width + dst_x);
    let last_x = ref_width as i64 - 1;
    let last_y = ref_height as i64 - 1;
    let filters = &SUBPEL_FILTERS[interp_filter as usize];

    // Fast path for unscaled border blocks (`x_step == 16 && y_step == 16`): edge-clamp each
    // needed source row into a tiny contiguous `u16` line buffer and reuse the fast contiguous
    // `madd_epi16` / copy kernels instead of `i32` gathers.
    if x_step == 16 && y_step == 16 {
        let x0 = x >> 4;
        let y0 = y >> 4;
        let fx = (x & 15) as usize;
        let fy = (y & 15) as usize;
        let is_bilinear = interp_filter == BILINEAR;
        let ref_base = ref_data.as_ptr();

        let fill_clamped_row = |ref_y: usize, col_start: i64, len: usize, out: *mut u16| {
            let row = ref_base.add(ref_y * ref_stride);
            if col_start >= 0 && col_start + (len as i64) - 1 <= last_x {
                std::ptr::copy_nonoverlapping(row.add(col_start as usize), out, len);
            } else {
                for i in 0..len {
                    let sx = (col_start + i as i64).clamp(0, last_x) as usize;
                    *out.add(i) = *row.add(sx);
                }
            }
        };

        let mut line = [MaybeUninit::<u16>::uninit(); MAX_BLOCK_DIM + 8];
        let line_ptr = line.as_mut_ptr() as *mut u16;

        match (fx == 0, fy == 0) {
            (true, true) => {
                for r in 0..h {
                    let ref_y = (y0 + r as i64).clamp(0, last_y) as usize;
                    fill_clamped_row(ref_y, x0, w, line_ptr);
                    copy_u16_avx2::<COMPOUND>(line_ptr, w, dst.add(r * dst_width), dst_width, w, 1);
                }
            }
            (false, true) => {
                for r in 0..h {
                    let ref_y = (y0 + r as i64).clamp(0, last_y) as usize;
                    fill_clamped_row(ref_y, x0 - 3, w + 7, line_ptr);
                    convolve_horiz_u16_avx2::<COMPOUND>(
                        line_ptr,
                        w + 7,
                        dst.add(r * dst_width),
                        dst_width,
                        w,
                        1,
                        &filters[fx],
                        is_bilinear,
                        max_val,
                    );
                }
            }
            (true, false) => {
                let mut tmp = [MaybeUninit::<u16>::uninit(); (MAX_BLOCK_DIM + 7) * MAX_BLOCK_DIM];
                let tmp_ptr = tmp.as_mut_ptr() as *mut u16;
                for r in 0..(h + 7) {
                    let ref_y = (y0 - 3 + r as i64).clamp(0, last_y) as usize;
                    fill_clamped_row(ref_y, x0, w, tmp_ptr.add(r * w));
                }
                convolve_vert_u16_avx2::<COMPOUND>(
                    tmp_ptr,
                    w,
                    dst,
                    dst_width,
                    w,
                    h,
                    &filters[fy],
                    is_bilinear,
                    max_val,
                );
            }
            (false, false) => {
                let mut tmp = [MaybeUninit::<u16>::uninit(); (MAX_BLOCK_DIM + 7) * MAX_BLOCK_DIM];
                let tmp_ptr = tmp.as_mut_ptr() as *mut u16;
                for r in 0..(h + 7) {
                    let ref_y = (y0 - 3 + r as i64).clamp(0, last_y) as usize;
                    fill_clamped_row(ref_y, x0 - 3, w + 7, line_ptr);
                    convolve_horiz_u16_avx2::<false>(
                        line_ptr,
                        w + 7,
                        tmp_ptr.add(r * w),
                        w,
                        w,
                        1,
                        &filters[fx],
                        is_bilinear,
                        max_val,
                    );
                }
                convolve_vert_u16_avx2::<COMPOUND>(
                    tmp_ptr,
                    w,
                    dst,
                    dst_width,
                    w,
                    h,
                    &filters[fy],
                    is_bilinear,
                    max_val,
                );
            }
        }
        return;
    }

    // General scaled reference path (SVC / resize): gather horizontal pass into uninitialized
    // `i32` stack scratch via raw pointer (never forming `&mut [i32]` over uninit), then pack
    // to `u16` into `dst` (with optional compound average).
    let mut pred = [MaybeUninit::<i32>::uninit(); MAX_BLOCK_DIM * MAX_BLOCK_DIM];
    let pred_ptr = pred.as_mut_ptr().cast::<i32>();
    block_inter_predict_scaled_strided_avx2(
        ref_data,
        ref_width,
        ref_stride,
        ref_height,
        x,
        y,
        x_step,
        y_step,
        w,
        h,
        intermediate_height,
        interp_filter,
        max_val,
        pred_ptr,
    );
    for r in 0..h {
        let sp = pred_ptr.add(r * w);
        let dp = dst.add(r * dst_width);
        if w == 4 {
            let v = _mm_loadu_si128(sp as *const __m128i);
            store_or_avg_4::<COMPOUND>(dp, _mm_packus_epi32(v, v));
        } else {
            let mut c = 0usize;
            while c < w {
                let lo = _mm_loadu_si128(sp.add(c) as *const __m128i);
                let hi = _mm_loadu_si128(sp.add(c + 4) as *const __m128i);
                store_or_avg_8::<COMPOUND>(dp.add(c), _mm_packus_epi32(lo, hi));
                c += 8;
            }
        }
    }
}

/// AVX2 mirror of `predict::block_inter_predict`'s scalar two-pass 8-tap FIR (legacy `i32`
/// output buffer entry point kept for `tests/unit/simd.rs`; decode paths call
/// [`block_inter_predict_u16_avx2`]).
///
/// # Safety
/// Same contract as before: caller confirms `avx2_enabled()` and that rows
/// `src_row0..src_row0 + intermediate_height` and cols `src_col0..src_col0 + w + 7` lie in
/// `ref_data` (stride `ref_width`).
#[cfg(test)]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_inter_predict_avx2(
    ref_data: &[u16],
    ref_width: usize,
    src_row0: i64,
    src_col0: i64,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    intermediate_height: usize,
    interp_filter: u8,
    max_val: i32,
    pred: &mut [i32],
) {
    debug_assert_eq!(w % 8, 0);
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert_eq!(pred.len(), w * h);
    debug_assert!(src_row0 >= 0 && src_col0 >= 0);

    let mut out_u16 = [MaybeUninit::<u16>::uninit(); MAX_BLOCK_DIM * MAX_BLOCK_DIM];
    let out_ptr = out_u16.as_mut_ptr().cast::<u16>();
    block_inter_predict_u16_ptr_avx2::<false>(
        ref_data,
        ref_width,
        out_ptr,
        w,
        src_col0 + 3,
        src_row0 + 3,
        fx,
        fy,
        w,
        h,
        interp_filter,
        max_val,
    );
    for (i, dst) in pred.iter_mut().enumerate() {
        *dst = *out_ptr.add(i) as i32;
    }
}

/// Width-4 companion to [`block_inter_predict_avx2`] (kept for `tests/unit/simd.rs`).
///
/// # Safety
/// Same contract as [`block_inter_predict_avx2`] with `w == 4`.
#[cfg(test)]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_inter_predict_avx2_w4(
    ref_data: &[u16],
    ref_width: usize,
    src_row0: i64,
    src_col0: i64,
    fx: usize,
    fy: usize,
    h: usize,
    intermediate_height: usize,
    interp_filter: u8,
    max_val: i32,
    pred: &mut [i32],
) {
    debug_assert!(h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert_eq!(pred.len(), 4 * h);
    debug_assert!(src_row0 >= 0 && src_col0 >= 0);

    let mut out_u16 = [MaybeUninit::<u16>::uninit(); MAX_BLOCK_DIM * 4];
    let out_ptr = out_u16.as_mut_ptr().cast::<u16>();
    block_inter_predict_u16_ptr_avx2::<false>(
        ref_data,
        ref_width,
        out_ptr,
        4,
        src_col0 + 3,
        src_row0 + 3,
        fx,
        fy,
        4,
        h,
        interp_filter,
        max_val,
    );
    for (i, dst) in pred.iter_mut().enumerate() {
        *dst = *out_ptr.add(i) as i32;
    }
}

/// General edge-clamping AVX2 mirror of `predict::block_inter_predict_scalar`'s two-pass
/// 8-tap subpel FIR (spec §8.5.2.3-8.5.2.4). Used for scaled references (`x_step`/`y_step`
/// differ from 16 -- SVC / resize) and directly tested by `tests/unit/simd.rs`.
///
/// # Safety
/// The caller must have confirmed `avx2_enabled()`, and must guarantee:
/// `ref_data.len() >= ref_width * ref_height`; `intermediate_height <= MAX_INTERMEDIATE_HEIGHT`;
/// horizontal source span `<= MAX_INTERMEDIATE_HEIGHT`; and `pred.len() == w * h`.
#[cfg(test)]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_inter_predict_scaled_avx2(
    ref_data: &[u16],
    ref_width: usize,
    ref_height: usize,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    intermediate_height: usize,
    interp_filter: u8,
    max_val: i32,
    pred: &mut [i32],
) {
    debug_assert_eq!(pred.len(), w * h);
    block_inter_predict_scaled_strided_avx2(
        ref_data,
        ref_width,
        ref_width,
        ref_height,
        x,
        y,
        x_step,
        y_step,
        w,
        h,
        intermediate_height,
        interp_filter,
        max_val,
        pred.as_mut_ptr(),
    );
}

/// Stride-aware general edge-clamping AVX2 inter-prediction core (`ref_stride >= ref_width`),
/// writing all `w * h` output samples to `pred_ptr` (allowing uninitialized stack scratch).
///
/// # Safety
/// Caller must confirm `avx2_enabled()`, `ref_stride >= ref_width`,
/// `ref_data.len() >= ref_stride * ref_height`, `intermediate_height <= MAX_INTERMEDIATE_HEIGHT`,
/// horizontal source span `<= MAX_INTERMEDIATE_HEIGHT`, and `pred_ptr` valid for writes of
/// `w * h` contiguous `i32` elements.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn block_inter_predict_scaled_strided_avx2(
    ref_data: &[u16],
    ref_width: usize,
    ref_stride: usize,
    ref_height: usize,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    intermediate_height: usize,
    interp_filter: u8,
    max_val: i32,
    pred_ptr: *mut i32,
) {
    debug_assert!(w == 4 || w.is_multiple_of(8));
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert!(ref_stride >= ref_width);
    debug_assert!(ref_data.len() >= ref_stride * ref_height);
    debug_assert!(x_step >= 0 && y_step >= 0);

    let filters = &SUBPEL_FILTERS[interp_filter as usize];
    let last_x = ref_width as i64 - 1;
    let last_y = ref_height as i64 - 1;

    let wp = if w == 4 { 8 } else { w };
    let groups = wp / 8;

    let col_min = (x >> 4) - 3;
    let span = ((((x & 15) + x_step * (w as i64 - 1)) >> 4) + 8) as usize;
    debug_assert!(span <= MAX_INTERMEDIATE_HEIGHT);
    let mut gidx = [[0i32; 8]; MAX_BLOCK_DIM / 8];
    let mut hcoef = [[[0i32; 8]; 8]; MAX_BLOCK_DIM / 8];
    for c in 0..wp {
        let p = x + x_step * (c.min(w - 1) as i64);
        gidx[c / 8][c % 8] = ((p >> 4) - 3 - col_min) as i32;
        for (t, &coeff) in filters[(p & 15) as usize].iter().enumerate() {
            hcoef[c / 8][t][c % 8] = coeff;
        }
    }

    let mut src_col = [0usize; MAX_INTERMEDIATE_HEIGHT];
    for (i, sc) in src_col[..span].iter_mut().enumerate() {
        *sc = (col_min + i as i64).clamp(0, last_x) as usize;
    }

    let round_add = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let max_clip = _mm256_set1_epi32(max_val);

    let mut intermediate = [MaybeUninit::<i32>::uninit(); MAX_INTERMEDIATE_HEIGHT * MAX_BLOCK_DIM];
    let inter_ptr = intermediate.as_mut_ptr() as *mut i32;
    let mut scratch = [MaybeUninit::<i32>::uninit(); MAX_INTERMEDIATE_HEIGHT];
    let scratch_ptr = scratch.as_mut_ptr() as *mut i32;

    for r in 0..intermediate_height {
        let ref_y = ((y >> 4) + r as i64 - 3).clamp(0, last_y) as usize;
        let row = &ref_data[ref_y * ref_stride..ref_y * ref_stride + ref_width];
        for (i, &sc) in src_col[..span].iter().enumerate() {
            *scratch_ptr.add(i) = row[sc] as i32;
        }
        for (g, (idx_lanes, taps)) in gidx[..groups]
            .iter()
            .zip(hcoef[..groups].iter())
            .enumerate()
        {
            let idx = _mm256_loadu_si256(idx_lanes.as_ptr() as *const __m256i);
            let mut acc = zero;
            for (t, tap_lanes) in taps.iter().enumerate() {
                let vals = _mm256_i32gather_epi32::<4>(scratch_ptr.add(t), idx);
                let coeff = _mm256_loadu_si256(tap_lanes.as_ptr() as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vals, coeff));
            }
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max_clip);
            _mm256_storeu_si256(inter_ptr.add(r * wp + g * 8) as *mut __m256i, acc);
        }
    }

    for r in 0..h {
        let p = (y & 15) + y_step * (r as i64);
        let vcoeffs = &filters[(p & 15) as usize];
        let base_row = (p >> 4) as usize;
        for g in 0..groups {
            let mut acc = zero;
            for (t, &tap) in vcoeffs.iter().enumerate() {
                let ptr = inter_ptr.add((base_row + t) * wp + g * 8);
                let vals = _mm256_loadu_si256(ptr as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vals, _mm256_set1_epi32(tap)));
            }
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max_clip);
            if w == 4 {
                _mm_storeu_si128(
                    pred_ptr.add(r * 4) as *mut __m128i,
                    _mm256_castsi256_si128(acc),
                );
            } else {
                _mm256_storeu_si256(pred_ptr.add(r * w + g * 8) as *mut __m256i, acc);
            }
        }
    }
}
