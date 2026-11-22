//! AVX2 SIMD mirrors of `predict::predict_intra`'s high-frequency modes (spec §8.5.1):
//! `V_PRED`, `H_PRED`, `DC_PRED` (all four neighbor-availability variants), and `TM_PRED`,
//! across every transform block size (`4x4`, `8x8`, `16x16`, `32x32`) and all VP9 bit depths
//! (`8`, `10`, `12` — every [`crate::framebuffer::Plane`] is `u16`-backed regardless of bit
//! depth). `predict.rs` owns the dispatch, neighbor gathering/clamping, and the scalar
//! fallback (which also handles the six directional modes `D45`..`D207`); this module owns
//! only the vector kernels, writing directly into the plane's raw `u16` buffer (skipping
//! the scalar path's 4 KB `i32` scratch array and per-pixel `Plane::set` loop).

use std::arch::x86_64::*;

/// Fills an `size x size` block at `(start_x, start_y)` in `plane_data` (row-major `u16`,
/// stride `plane_width`) with the constant `val`. Shared by `DC_PRED`, unavailable-above
/// `V_PRED`, and unavailable-left `H_PRED`.
///
/// # Safety
/// `avx2_enabled()` must hold; `size` must be one of `4, 8, 16, 32`; and rows
/// `start_y..start_y + size` x columns `start_x..start_x + size` must be in bounds for
/// `plane_data`.
#[target_feature(enable = "avx2")]
pub unsafe fn intra_pred_fill_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    val: u16,
    size: usize,
) {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    debug_assert!(start_x + size <= plane_width);
    debug_assert!((start_y + size - 1) * plane_width + start_x + size <= plane_data.len());

    let base = plane_data.as_mut_ptr();
    match size {
        4 => {
            let v = _mm_set1_epi16(val as i16);
            for i in 0..4 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storel_epi64(row as *mut __m128i, v);
            }
        }
        8 => {
            let v = _mm_set1_epi16(val as i16);
            for i in 0..8 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storeu_si128(row as *mut __m128i, v);
            }
        }
        16 => {
            let v = _mm256_set1_epi16(val as i16);
            for i in 0..16 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, v);
            }
        }
        32 => {
            let v = _mm256_set1_epi16(val as i16);
            for i in 0..32 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, v);
                _mm256_storeu_si256(row.add(16) as *mut __m256i, v);
            }
        }
        _ => unreachable!("intra_pred_fill_avx2: invalid block size {size}"),
    }
}

/// AVX2 `V_PRED` (vertical prediction, spec §8.5.1): copies `above[0..size]` into every
/// row of the `size x size` destination block at `(start_x, start_y)`. Bit-depth-agnostic
/// (`u16` copy; exact for 8/10/12-bit).
///
/// # Safety
/// `avx2_enabled()` must hold; `size` must be one of `4, 8, 16, 32`; `above.len() >= size`;
/// and rows `start_y..start_y + size` x columns `start_x..start_x + size` must be in bounds
/// for `plane_data`.
#[target_feature(enable = "avx2")]
pub unsafe fn intra_pred_v_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    above: &[u16],
    size: usize,
) {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    debug_assert!(above.len() >= size);
    debug_assert!(start_x + size <= plane_width);
    debug_assert!((start_y + size - 1) * plane_width + start_x + size <= plane_data.len());

    let base = plane_data.as_mut_ptr();
    let src = above.as_ptr();
    match size {
        4 => {
            let v = _mm_loadl_epi64(src as *const __m128i);
            for i in 0..4 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storel_epi64(row as *mut __m128i, v);
            }
        }
        8 => {
            let v = _mm_loadu_si128(src as *const __m128i);
            for i in 0..8 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storeu_si128(row as *mut __m128i, v);
            }
        }
        16 => {
            let v = _mm256_loadu_si256(src as *const __m256i);
            for i in 0..16 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, v);
            }
        }
        32 => {
            let v0 = _mm256_loadu_si256(src as *const __m256i);
            let v1 = _mm256_loadu_si256(src.add(16) as *const __m256i);
            for i in 0..32 {
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, v0);
                _mm256_storeu_si256(row.add(16) as *mut __m256i, v1);
            }
        }
        _ => unreachable!("intra_pred_v_avx2: invalid block size {size}"),
    }
}

/// AVX2 `H_PRED` (horizontal prediction, spec §8.5.1): broadcasts `left[i]` across row `i`
/// of the `size x size` destination block at `(start_x, start_y)`. Bit-depth-agnostic
/// (`u16` broadcast; exact for 8/10/12-bit).
///
/// # Safety
/// `avx2_enabled()` must hold; `size` must be one of `4, 8, 16, 32`; `left.len() >= size`;
/// and rows `start_y..start_y + size` x columns `start_x..start_x + size` must be in bounds
/// for `plane_data`.
#[target_feature(enable = "avx2")]
pub unsafe fn intra_pred_h_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    left: &[u16],
    size: usize,
) {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    debug_assert!(left.len() >= size);
    debug_assert!(start_x + size <= plane_width);
    debug_assert!((start_y + size - 1) * plane_width + start_x + size <= plane_data.len());

    let base = plane_data.as_mut_ptr();
    match size {
        4 => {
            for (i, &lv) in left[..4].iter().enumerate() {
                let v = _mm_set1_epi16(lv as i16);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storel_epi64(row as *mut __m128i, v);
            }
        }
        8 => {
            for (i, &lv) in left[..8].iter().enumerate() {
                let v = _mm_set1_epi16(lv as i16);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storeu_si128(row as *mut __m128i, v);
            }
        }
        16 => {
            for (i, &lv) in left[..16].iter().enumerate() {
                let v = _mm256_set1_epi16(lv as i16);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, v);
            }
        }
        32 => {
            for (i, &lv) in left[..32].iter().enumerate() {
                let v = _mm256_set1_epi16(lv as i16);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, v);
                _mm256_storeu_si256(row.add(16) as *mut __m256i, v);
            }
        }
        _ => unreachable!("intra_pred_h_avx2: invalid block size {size}"),
    }
}

/// Horizontal sum of `size` (`4, 8, 16, 32`) `u16` VP9 samples (each `<= 4095 < 32768`, so
/// non-negative as `i16`) via pairwise `madd_epi16(., 1)` -> `i32` reduction. Exact for
/// every 8/10/12-bit VP9 sample (max sum `32 * 4095 = 131_040`, far below `i32::MAX`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sum_u16_avx2(samples: &[u16], size: usize) -> i32 {
    debug_assert!(samples.len() >= size);
    debug_assert!(samples[..size].iter().all(|&v| v <= 4095));

    let ptr = samples.as_ptr();
    match size {
        4 => {
            let v = _mm_loadl_epi64(ptr as *const __m128i);
            let pairs = _mm_madd_epi16(v, _mm_set1_epi16(1));
            let s = _mm_add_epi32(pairs, _mm_srli_si128(pairs, 4));
            _mm_cvtsi128_si32(s)
        }
        8 => {
            let v = _mm_loadu_si128(ptr as *const __m128i);
            let pairs = _mm_madd_epi16(v, _mm_set1_epi16(1));
            let s2 = _mm_add_epi32(pairs, _mm_srli_si128(pairs, 8));
            let s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
            _mm_cvtsi128_si32(s1)
        }
        16 => {
            let v = _mm256_loadu_si256(ptr as *const __m256i);
            let pairs = _mm256_madd_epi16(v, _mm256_set1_epi16(1));
            let s128 = _mm_add_epi32(
                _mm256_castsi256_si128(pairs),
                _mm256_extracti128_si256(pairs, 1),
            );
            let s2 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
            let s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
            _mm_cvtsi128_si32(s1)
        }
        32 => {
            let ones = _mm256_set1_epi16(1);
            let v0 = _mm256_loadu_si256(ptr as *const __m256i);
            let v1 = _mm256_loadu_si256(ptr.add(16) as *const __m256i);
            let pairs = _mm256_add_epi32(_mm256_madd_epi16(v0, ones), _mm256_madd_epi16(v1, ones));
            let s128 = _mm_add_epi32(
                _mm256_castsi256_si128(pairs),
                _mm256_extracti128_si256(pairs, 1),
            );
            let s2 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
            let s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
            _mm_cvtsi128_si32(s1)
        }
        _ => unreachable!("sum_u16_avx2: invalid block size {size}"),
    }
}

/// AVX2 `DC_PRED` (spec §8.5.1): handles all four neighbor-availability variants
/// (`above: Some/None` x `left: Some/None`) across sizes `4/8/16/32` (`log2_size` in
/// `2..=5`) and all VP9 bit depths (`8/10/12`), reducing the available neighbors in `i32`
/// and filling the destination block via [`intra_pred_fill_avx2`].
///
/// # Safety
/// `avx2_enabled()` must hold; `log2_size` must be in `2..=5` (`size = 1 << log2_size`);
/// any `Some(slice)` must have length `>= size` with samples in `0..=4095`; and rows
/// `start_y..start_y + size` x columns `start_x..start_x + size` must be in bounds for
/// `plane_data`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub unsafe fn intra_pred_dc_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    above: Option<&[u16]>,
    left: Option<&[u16]>,
    log2_size: u32,
    bit_depth: u8,
) {
    debug_assert!((2..=5).contains(&log2_size));
    let size = 1usize << log2_size;
    let dc = match (above, left) {
        (Some(a), Some(l)) => {
            let sum = sum_u16_avx2(a, size) + sum_u16_avx2(l, size);
            (sum + size as i32) >> (log2_size + 1)
        }
        (None, Some(l)) => {
            let sum = sum_u16_avx2(l, size);
            (sum + (1 << (log2_size - 1))) >> log2_size
        }
        (Some(a), None) => {
            let sum = sum_u16_avx2(a, size);
            (sum + (1 << (log2_size - 1))) >> log2_size
        }
        (None, None) => 1i32 << (bit_depth - 1),
    };
    intra_pred_fill_avx2(plane_data, plane_width, start_x, start_y, dc as u16, size);
}

/// AVX2 `TM_PRED` (TrueMotion prediction, spec §8.5.1): writes
/// `clip(above[j] + left[i] - top_left, 0, (1 << bit_depth) - 1)` directly into the
/// `size x size` block at `(start_x, start_y)`.
///
/// Exact in signed `i16` lanes for ALL VP9 bit depths (`8/10/12`): every input (`above[j]`,
/// `left[i]`, `top_left`) is in `0..=4095`, so `above[j] - top_left` (precomputed once per
/// block before the row loop) lies in `-4095..=4095`, and adding `left[i]` stays in
/// `-4095..=8190` — strictly inside `i16` (`-32768..=32767`) with zero wrap. Signed
/// `max(0)` / `min(max_val)` then leaves a non-negative value `0..=max_val` directly
/// storable as `u16`.
///
/// # Safety
/// `avx2_enabled()` must hold; `size` must be one of `4, 8, 16, 32`; `above.len() >= size`
/// and `left.len() >= size` with all samples (and `top_left`) in `0..=(1 << bit_depth) - 1`
/// (at most `4095`); and rows `start_y..start_y + size` x columns
/// `start_x..start_x + size` must be in bounds for `plane_data`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub unsafe fn intra_pred_tm_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    above: &[u16],
    left: &[u16],
    top_left: u16,
    size: usize,
    bit_depth: u8,
) {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    debug_assert!(above.len() >= size && left.len() >= size);
    debug_assert!(start_x + size <= plane_width);
    debug_assert!((start_y + size - 1) * plane_width + start_x + size <= plane_data.len());
    debug_assert!(top_left <= 4095);
    debug_assert!(above[..size].iter().all(|&v| v <= 4095));
    debug_assert!(left[..size].iter().all(|&v| v <= 4095));

    let base = plane_data.as_mut_ptr();
    let max_val = ((1i32 << bit_depth) - 1) as i16;
    match size {
        4 => {
            let zero = _mm_setzero_si128();
            let max_v = _mm_set1_epi16(max_val);
            let tl = _mm_set1_epi16(top_left as i16);
            let diff = _mm_sub_epi16(_mm_loadl_epi64(above.as_ptr() as *const __m128i), tl);
            for (i, &lv) in left[..4].iter().enumerate() {
                let sum = _mm_add_epi16(diff, _mm_set1_epi16(lv as i16));
                let clipped = _mm_min_epi16(_mm_max_epi16(sum, zero), max_v);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storel_epi64(row as *mut __m128i, clipped);
            }
        }
        8 => {
            let zero = _mm_setzero_si128();
            let max_v = _mm_set1_epi16(max_val);
            let tl = _mm_set1_epi16(top_left as i16);
            let diff = _mm_sub_epi16(_mm_loadu_si128(above.as_ptr() as *const __m128i), tl);
            for (i, &lv) in left[..8].iter().enumerate() {
                let sum = _mm_add_epi16(diff, _mm_set1_epi16(lv as i16));
                let clipped = _mm_min_epi16(_mm_max_epi16(sum, zero), max_v);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm_storeu_si128(row as *mut __m128i, clipped);
            }
        }
        16 => {
            let zero = _mm256_setzero_si256();
            let max_v = _mm256_set1_epi16(max_val);
            let tl = _mm256_set1_epi16(top_left as i16);
            let diff = _mm256_sub_epi16(_mm256_loadu_si256(above.as_ptr() as *const __m256i), tl);
            for (i, &lv) in left[..16].iter().enumerate() {
                let sum = _mm256_add_epi16(diff, _mm256_set1_epi16(lv as i16));
                let clipped = _mm256_min_epi16(_mm256_max_epi16(sum, zero), max_v);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, clipped);
            }
        }
        32 => {
            let zero = _mm256_setzero_si256();
            let max_v = _mm256_set1_epi16(max_val);
            let tl = _mm256_set1_epi16(top_left as i16);
            let diff0 = _mm256_sub_epi16(_mm256_loadu_si256(above.as_ptr() as *const __m256i), tl);
            let diff1 = _mm256_sub_epi16(
                _mm256_loadu_si256(above.as_ptr().add(16) as *const __m256i),
                tl,
            );
            for (i, &lv) in left[..32].iter().enumerate() {
                let lvec = _mm256_set1_epi16(lv as i16);
                let c0 =
                    _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(diff0, lvec), zero), max_v);
                let c1 =
                    _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(diff1, lvec), zero), max_v);
                let row = base.add((start_y + i) * plane_width + start_x);
                _mm256_storeu_si256(row as *mut __m256i, c0);
                _mm256_storeu_si256(row.add(16) as *mut __m256i, c1);
            }
        }
        _ => unreachable!("intra_pred_tm_avx2: invalid block size {size}"),
    }
}
