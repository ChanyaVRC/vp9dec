//! AVX2 SIMD mirror of `predict::block_inter_predict`'s scalar two-pass 8-tap subpel
//! convolution (spec §8.5.2.4), for the UNSCALED case only (`x_step == y_step == 16`,
//! i.e. reference frame same size as the current frame -- see
//! docs/implementation-notes.md "SIMD wave 2"). `predict.rs` owns the scaled path and the
//! scalar fallback; this module owns only the vector kernel.
//!
//! x86_64-only for now (`core::arch::x86_64` intrinsics, zero dependencies). A NEON
//! mirror for aarch64 would be a sibling `#[cfg(target_arch = "aarch64")]` module behind
//! the same `predict.rs` dispatch point -- not implemented this wave.

use crate::predict::{MAX_BLOCK_DIM, MAX_INTERMEDIATE_HEIGHT};
use crate::subpel::SUBPEL_FILTERS;
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Whether the AVX2 fast path should be used: the CPU supports AVX2 and it hasn't been
/// force-disabled. Cached (the feature/env probe cost isn't worth paying per block --
/// `predict::block_inter_predict` calls this once per inter-predicted plane block).
///
/// `VP9DEC_NO_SIMD` (any value, checked once) forces the scalar path even on an
/// AVX2-capable machine -- the wave-2 verification hook to exercise the fallback path
/// (see docs/implementation-notes.md "SIMD wave 2"): the official/ffmpeg-cross-decode
/// sweeps are run once with this unset (SIMD path) and once with it set (scalar path),
/// both expected to pass, to prove the two agree.
pub fn avx2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("VP9DEC_NO_SIMD").is_none() && is_x86_feature_detected!("avx2")
    })
}

/// AVX2 mirror of `predict::block_inter_predict`'s scalar two-pass 8-tap FIR.
///
/// Only valid for the unscaled case (`x_step == y_step == 16`), which lets the caller
/// simplify the scalar loop's per-column `filters[(p & 15) as usize]` lookup and
/// `(p >> 4)` row/column arithmetic down to a single filter (`fx`/`fy`) and a flat
/// `src_row0`/`src_col0` origin -- see `predict.rs`'s call site for the derivation
/// (`p & 15` and `p >> 4 - c` are both step-invariant when `step == 16`).
///
/// `w` must be a multiple of 8 (this processes 8 output columns per AVX2 lane group;
/// width-4 blocks stay on the scalar path). `h <= MAX_BLOCK_DIM`,
/// `intermediate_height <= MAX_INTERMEDIATE_HEIGHT`, `bit_depth == 8` (the only
/// supported depth -- see `lib.rs`'s `UnsupportedBitDepth` rejection).
///
/// # Safety
/// The caller must have confirmed `avx2_enabled()` (this fn requires the `avx2` target
/// feature to be genuinely available at runtime) and must guarantee that every source
/// pixel this call reads -- rows `src_row0..src_row0 + intermediate_height` and columns
/// `src_col0..src_col0 + w + 7` of `ref_data` (row-major, stride `ref_width`) -- lies
/// within `ref_data`'s bounds. Unlike the scalar path, this function does **not**
/// clamp/border-replicate: reproducing that with AVX2 would need a byte gather (x86 has
/// none), so the caller instead falls back to the scalar path whenever any of those
/// reads would land outside `ref_data` (only near reference-frame edges -- see
/// `predict.rs`'s `in_bounds` check).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_inter_predict_avx2(
    ref_data: &[u8],
    ref_width: usize,
    src_row0: i64,
    src_col0: i64,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    intermediate_height: usize,
    interp_filter: u8,
    pred: &mut [i32],
) {
    debug_assert_eq!(w % 8, 0);
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert_eq!(pred.len(), w * h);
    debug_assert!(src_row0 >= 0 && src_col0 >= 0);
    debug_assert!(
        ((src_row0 as usize) + intermediate_height - 1) * ref_width + (src_col0 as usize) + w + 6
            < ref_data.len()
    );

    let filters = &SUBPEL_FILTERS[interp_filter as usize];
    let hcoeffs = &filters[fx];
    let vcoeffs = &filters[fy];

    let round_add = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let max255 = _mm256_set1_epi32(255);

    // Same fixed-size scratch as the scalar path's `intermediate`: the spec's two-pass
    // structure needs the full horizontal-filter output (for rows both above and below
    // any given output row) before the vertical pass can run.
    let mut intermediate = [0i32; MAX_INTERMEDIATE_HEIGHT * MAX_BLOCK_DIM];

    // Horizontal pass. `fx` is the one filter used for every row/column here (the
    // x_step==16 simplification), and `ref_y = src_row0 + r` / `ref_x = src_col0 + c + t`
    // need no clamp -- the caller (predict.rs) already proved every such read is
    // in-bounds.
    let ref_base = ref_data.as_ptr();
    for r in 0..intermediate_height {
        let row_ptr = ref_base.add(((src_row0 as usize) + r) * ref_width + src_col0 as usize);
        let mut c = 0usize;
        while c < w {
            let mut acc = zero;
            for (t, &tap) in hcoeffs.iter().enumerate() {
                // 8 bytes -> 8 x i32 (zero-extend, matching the scalar `.get(..) as i32`
                // on a u8 sample); reads exactly the columns this chunk's 8 outputs need
                // for this tap, no more (see the Safety section's bound).
                let bytes = _mm_loadl_epi64(row_ptr.add(c + t) as *const __m128i);
                let widened = _mm256_cvtepu8_epi32(bytes);
                let coeff = _mm256_set1_epi32(tap);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(widened, coeff));
            }
            // round2(s, 7) then clip1 (bit_depth == 8, so Clip3(0, 255, .)) -- identical
            // arithmetic and accumulation order to the scalar `round2`/`clip1`, just 8
            // lanes at once.
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max255);
            _mm256_storeu_si256(
                intermediate.as_mut_ptr().add(r * w + c) as *mut __m256i,
                acc,
            );
            c += 8;
        }
    }

    // Vertical pass. `fy` is the one filter used for every output row (same
    // simplification), and `base_row == r` exactly (the y_step==16 case of the scalar
    // loop's `p >> 4`), so tap `t` reads intermediate row `r + t` directly.
    for r in 0..h {
        let mut c = 0usize;
        while c < w {
            let mut acc = zero;
            for (t, &tap) in vcoeffs.iter().enumerate() {
                let p = intermediate.as_ptr().add((r + t) * w + c);
                let vals = _mm256_loadu_si256(p as *const __m256i);
                let coeff = _mm256_set1_epi32(tap);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vals, coeff));
            }
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max255);
            _mm256_storeu_si256(pred.as_mut_ptr().add(r * w + c) as *mut __m256i, acc);
            c += 8;
        }
    }
}

/// AVX2 mirror of `loop_filter.rs`'s narrow (spec §8.8.5.2, `filter4`) and wide 8-tap
/// (§8.8.5.3, `log2_size == 3`) deblocking filters, applied to 8 contiguous along-edge
/// positions on a HORIZONTAL edge (loop_filter.rs pass==1: taps run in the row direction,
/// i.e. `dx=0,dy=1` in that file's terms) -- see docs/implementation-notes.md "SIMD wave 3".
/// Vertical edges (pass==0) and the rarer 16-tap "wide2" filter (`TX_16X16` positions) are
/// not handled here; `loop_filter.rs`'s dispatch keeps those scalar.
///
/// `plane_data`/`plane_width` is the raw row-major plane buffer (stride == `plane_width`).
/// `(x0, y0)` is lane 0's position (loop_filter.rs's `sample_filtering(x, y, ...)` for the
/// first of the 8 lanes); the other 7 lanes are the next 7 contiguous columns
/// `x0+1..=x0+7` at the same row `y0` (this orientation's along-edge axis -- see the
/// `debug_assert_eq!`s at the call site proving contiguity).
///
/// `eligible[lane]` / `is_tx8[lane]` are 0/-1 masks (not `bool`, so the caller can build them
/// once and this fn just loads them): `eligible` gates whether the lane is written at all
/// (false = leave the plane untouched -- inactive lane, or a `TX_16X16` lane the caller
/// already ran through the scalar wide2 path directly); `is_tx8` is whether `filter_size ==
/// TX_8X8` (true) vs `TX_4X4` (false) -- `TX_16X16` lanes must never reach this fn, since the
/// kernel never reads `p4..p7`/`q4..q7` and so can only ever pick narrow or wide8, exactly
/// mirroring `loop_filter.rs::sample_filtering`'s three-way branch restricted to those two
/// arms. `limit`/`blimit`/`thresh` are each lane's spec §8.8.4 filter strength (only read for
/// eligible lanes).
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and that reading columns `x0..x0+8` at rows
/// `y0-4..=y0+3` (the `p3..q3` window spec §8.8.5.1's `compute_filter_mask` reads) is within
/// `plane_data`'s bounds for every lane -- true whenever at least one lane is eligible, by
/// the same reasoning the existing scalar path already relies on (see the call site's
/// comment).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn loop_filter_horiz8_avx2(
    plane_data: &mut [u8],
    plane_width: usize,
    x0: usize,
    y0: usize,
    eligible: &[i32; 8],
    is_tx8: &[i32; 8],
    limit: &[i32; 8],
    blimit: &[i32; 8],
    thresh: &[i32; 8],
) {
    let base = plane_data.as_mut_ptr();

    // Loads 8 contiguous bytes (columns x0..x0+8) from row `y0 + dy`, widened to 8xi32 --
    // one AVX2 lane per along-edge position, matching `get_off`'s `plane.get(x, y+k)` for
    // pass==1 (`dx=0,dy=1`).
    let load_row = |dy: i64| -> __m256i {
        let row_ptr = base.offset(((y0 as i64 + dy) * plane_width as i64 + x0 as i64) as isize);
        _mm256_cvtepu8_epi32(_mm_loadl_epi64(row_ptr as *const __m128i))
    };
    let p3 = load_row(-4);
    let p2 = load_row(-3);
    let p1 = load_row(-2);
    let p0 = load_row(-1);
    let q0 = load_row(0);
    let q1 = load_row(1);
    let q2 = load_row(2);
    let q3 = load_row(3);

    let abs_diff = |a: __m256i, b: __m256i| _mm256_abs_epi32(_mm256_sub_epi32(a, b));
    let gt = |a: __m256i, b: __m256i| _mm256_cmpgt_epi32(a, b);
    let all_ones = _mm256_set1_epi32(-1);
    let not_mask = |m: __m256i| _mm256_xor_si256(m, all_ones);

    let limit_v = _mm256_loadu_si256(limit.as_ptr() as *const __m256i);
    let blimit_v = _mm256_loadu_si256(blimit.as_ptr() as *const __m256i);
    let thresh_v = _mm256_loadu_si256(thresh.as_ptr() as *const __m256i);
    let eligible_v = _mm256_loadu_si256(eligible.as_ptr() as *const __m256i);
    let is_tx8_v = _mm256_loadu_si256(is_tx8.as_ptr() as *const __m256i);

    // Spec §8.8.5.1 "Filter mask process".
    let hev_mask = _mm256_or_si256(
        gt(abs_diff(p1, p0), thresh_v),
        gt(abs_diff(q1, q0), thresh_v),
    );

    let mut mask = gt(abs_diff(p3, p2), limit_v);
    mask = _mm256_or_si256(mask, gt(abs_diff(p2, p1), limit_v));
    mask = _mm256_or_si256(mask, gt(abs_diff(p1, p0), limit_v));
    mask = _mm256_or_si256(mask, gt(abs_diff(q1, q0), limit_v));
    mask = _mm256_or_si256(mask, gt(abs_diff(q2, q1), limit_v));
    mask = _mm256_or_si256(mask, gt(abs_diff(q3, q2), limit_v));
    let blimit_sum = _mm256_add_epi32(
        _mm256_mullo_epi32(abs_diff(p0, q0), _mm256_set1_epi32(2)),
        _mm256_srli_epi32(abs_diff(p1, q1), 1),
    );
    mask = _mm256_or_si256(mask, gt(blimit_sum, blimit_v));
    let filter_mask = not_mask(mask);
    let elig_and_fm = _mm256_and_si256(eligible_v, filter_mask);

    // Fast path: if no lane in this batch has filter_size == TX_8X8 (the common case for
    // detailed content, where most edges are TX_4X4), wide8 (the "flat" filter) can never be
    // selected -- selection requires `flat_mask & is_tx8`, and `is_tx8_v` is all-zero -- so
    // skip computing flat_mask (6 abs-diffs) and the wide8 weighted sums (6 more) entirely,
    // and skip writing p2/q2 (only wide8 ever touches them). `_mm256_testz_si256(a, a)` is a
    // single-instruction "is `a` all-zero" test.
    let no_tx8 = _mm256_testz_si256(is_tx8_v, is_tx8_v) != 0;

    // Spec §8.8.5.2 "Narrow filter process" (`filter4`).
    let c128 = _mm256_set1_epi32(128);
    let clamp4 = |v: __m256i| {
        _mm256_min_epi32(
            _mm256_max_epi32(v, _mm256_set1_epi32(-128)),
            _mm256_set1_epi32(127),
        )
    };

    let ps1 = _mm256_sub_epi32(p1, c128);
    let ps0 = _mm256_sub_epi32(p0, c128);
    let qs0 = _mm256_sub_epi32(q0, c128);
    let qs1 = _mm256_sub_epi32(q1, c128);

    let filt_hev = clamp4(_mm256_sub_epi32(ps1, qs1));
    let filt_base = _mm256_blendv_epi8(_mm256_setzero_si256(), filt_hev, hev_mask);
    let filt = clamp4(_mm256_add_epi32(
        filt_base,
        _mm256_mullo_epi32(_mm256_sub_epi32(qs0, ps0), _mm256_set1_epi32(3)),
    ));
    let filter1 = _mm256_srai_epi32(clamp4(_mm256_add_epi32(filt, _mm256_set1_epi32(4))), 3);
    let filter2 = _mm256_srai_epi32(clamp4(_mm256_add_epi32(filt, _mm256_set1_epi32(3))), 3);

    let oq0_narrow = _mm256_add_epi32(clamp4(_mm256_sub_epi32(qs0, filter1)), c128);
    let op0_narrow = _mm256_add_epi32(clamp4(_mm256_add_epi32(ps0, filter2)), c128);

    let round_f = _mm256_srai_epi32(_mm256_add_epi32(filter1, _mm256_set1_epi32(1)), 1);
    let oq1_narrow = _mm256_blendv_epi8(
        _mm256_add_epi32(clamp4(_mm256_sub_epi32(qs1, round_f)), c128),
        q1,
        hev_mask,
    );
    let op1_narrow = _mm256_blendv_epi8(
        _mm256_add_epi32(clamp4(_mm256_add_epi32(ps1, round_f)), c128),
        p1,
        hev_mask,
    );

    // Packs 8xi32 (each guaranteed in 0..=255: narrow's clamp4(..)+128 and wide8's
    // round2(sum-of-8-pixel-values, 3) can't leave that range) down to 8 contiguous bytes
    // and stores at row `y0 + dy`, columns x0..x0+8 -- the exact reverse of `load_row`.
    let store_row = |dy: i64, v: __m256i| {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let u16x8 = _mm_packus_epi32(lo, hi);
        let u8x16 = _mm_packus_epi16(u16x8, u16x8);
        let row_ptr = base.offset(((y0 as i64 + dy) * plane_width as i64 + x0 as i64) as isize);
        _mm_storel_epi64(row_ptr as *mut __m128i, u8x16);
    };

    if no_tx8 {
        // Fast path (see `no_tx8`'s doc above): wide8 can't be selected, so `use_narrow ==
        // elig_and_fm` and p2/q2 are never touched -- skip computing/storing them.
        let p1_out = _mm256_blendv_epi8(p1, op1_narrow, elig_and_fm);
        let p0_out = _mm256_blendv_epi8(p0, op0_narrow, elig_and_fm);
        let q0_out = _mm256_blendv_epi8(q0, oq0_narrow, elig_and_fm);
        let q1_out = _mm256_blendv_epi8(q1, oq1_narrow, elig_and_fm);
        store_row(-2, p1_out);
        store_row(-1, p0_out);
        store_row(0, q0_out);
        store_row(1, q1_out);
        return;
    }

    // flat_mask, gated on filter_size >= TX_8X8 (is_tx8_v) exactly like the scalar
    // `if filter_size >= TX_8X8 { .. }` (else flat_mask stays false).
    let one = _mm256_set1_epi32(1);
    let mut fm = gt(abs_diff(p1, p0), one);
    fm = _mm256_or_si256(fm, gt(abs_diff(q1, q0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(p2, p0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(q2, q0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(p3, p0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(q3, q0), one));
    let flat_mask = _mm256_and_si256(not_mask(fm), is_tx8_v);

    let use_wide8 = _mm256_and_si256(elig_and_fm, flat_mask);
    let use_narrow = _mm256_andnot_si256(flat_mask, elig_and_fm);

    // Spec §8.8.5.3 "Wide filter process" (log2_size == 3, the 8-tap "flat" filter).
    // Closed-form weighted sums equivalent to the spec's O(n^2) loop (verified by hand
    // against the spec's `t = get_off(i) + sum_j get_off(clip3(-4,3,i+j))` derivation --
    // reordering the same set of integer terms is exact, not just "close").
    let round2_3 = |t: __m256i| _mm256_srai_epi32(_mm256_add_epi32(t, _mm256_set1_epi32(4)), 3);
    let mul = |v: __m256i, c: i32| _mm256_mullo_epi32(v, _mm256_set1_epi32(c));
    let add = _mm256_add_epi32;
    let sum5 = |a: __m256i, b: __m256i, c: __m256i, d: __m256i, e: __m256i| {
        add(add(add(a, b), add(c, d)), e)
    };
    let sum6 = |a: __m256i, b: __m256i, c: __m256i, d: __m256i, e: __m256i, f: __m256i| {
        add(sum5(a, b, c, d, e), f)
    };
    let sum7 =
        |a: __m256i, b: __m256i, c: __m256i, d: __m256i, e: __m256i, f: __m256i, g: __m256i| {
            add(sum6(a, b, c, d, e, f), g)
        };

    let op2_wide = round2_3(sum5(mul(p3, 3), mul(p2, 2), p1, p0, q0));
    let op1_wide = round2_3(sum6(mul(p3, 2), p2, mul(p1, 2), p0, q0, q1));
    let op0_wide = round2_3(sum7(p3, p2, p1, mul(p0, 2), q0, q1, q2));
    let oq0_wide = round2_3(sum7(p2, p1, p0, mul(q0, 2), q1, q2, q3));
    let oq1_wide = round2_3(sum6(p1, p0, q0, mul(q1, 2), q2, mul(q3, 2)));
    let oq2_wide = round2_3(sum5(p0, q0, q1, mul(q2, 2), mul(q3, 3)));

    // Blend: original (untouched) -> narrow candidate where selected -> wide8 candidate
    // where selected. `use_narrow`/`use_wide8` are mutually exclusive, so order doesn't
    // matter for the overlap; positions the winning filter doesn't touch (p2/q2 for narrow,
    // p3/q3 for both) fall through to their own candidate/original as appropriate.
    let blend3 = |orig: __m256i, narrow: __m256i, wide: __m256i| {
        _mm256_blendv_epi8(
            _mm256_blendv_epi8(orig, narrow, use_narrow),
            wide,
            use_wide8,
        )
    };
    let p2_out = _mm256_blendv_epi8(p2, op2_wide, use_wide8);
    let p1_out = blend3(p1, op1_narrow, op1_wide);
    let p0_out = blend3(p0, op0_narrow, op0_wide);
    let q0_out = blend3(q0, oq0_narrow, oq0_wide);
    let q1_out = blend3(q1, oq1_narrow, oq1_wide);
    let q2_out = _mm256_blendv_epi8(q2, oq2_wide, use_wide8);

    store_row(-3, p2_out);
    store_row(-2, p1_out);
    store_row(-1, p0_out);
    store_row(0, q0_out);
    store_row(1, q1_out);
    store_row(2, q2_out);
}
