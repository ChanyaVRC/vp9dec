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
/// `intermediate_height <= MAX_INTERMEDIATE_HEIGHT`, `bit_depth == 8` (the caller only
/// dispatches here for 8-bit frames -- see `predict.rs`'s call site). `ref_data` is
/// `Plane::as_slice()`'s `u16` buffer (every plane is `u16`-backed regardless of bit
/// depth, see `framebuffer.rs`); at `bit_depth == 8` every sample is known to fit in
/// `0..=255`, so widening `u16 -> i32` here is bit-exact with the scalar path's
/// `u16 as i32`.
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
                // 8 u16 samples -> 8 x i32 (zero-extend, matching the scalar `.get(..) as
                // i32` on a u16 sample); reads exactly the columns this chunk's 8 outputs
                // need for this tap, no more (see the Safety section's bound).
                let vals = _mm_loadu_si128(row_ptr.add(c + t) as *const __m128i);
                let widened = _mm256_cvtepu16_epi32(vals);
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

/// Width-4 companion to [`block_inter_predict_avx2`]: the same UNSCALED 8-tap two-pass subpel
/// convolution, but 4 output columns per row (a natural 128-bit `_mm_*` lane group) instead of
/// 8. Width-4 blocks (the `4x4` / `4x8` partitions) are not a multiple of the main kernel's
/// 8-wide group, so they had stayed on the scalar path. Same arithmetic, order and clipping as
/// the main kernel (so bit-exact with it and the scalar loop), just 4 lanes wide.
///
/// # Safety
/// Same contract as [`block_inter_predict_avx2`] with `w == 4`: caller must have confirmed
/// `avx2_enabled()` and that every source pixel this reads -- rows
/// `src_row0..src_row0 + intermediate_height` and columns `src_col0..src_col0 + 4 + 7` of
/// `ref_data` (row-major, stride `ref_width`) -- is within bounds (the caller's `in_bounds`
/// check covers this, falling back to the scalar path near reference-frame edges).
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
    pred: &mut [i32],
) {
    debug_assert!(h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert_eq!(pred.len(), 4 * h);
    debug_assert!(src_row0 >= 0 && src_col0 >= 0);
    debug_assert!(
        ((src_row0 as usize) + intermediate_height - 1) * ref_width + (src_col0 as usize) + 4 + 6
            < ref_data.len()
    );

    let filters = &SUBPEL_FILTERS[interp_filter as usize];
    let hcoeffs = &filters[fx];
    let vcoeffs = &filters[fy];

    let round_add = _mm_set1_epi32(64);
    let zero = _mm_setzero_si128();
    let max255 = _mm_set1_epi32(255);

    // Stride-4 scratch (only the first `intermediate_height * 4` entries are used); the same
    // two-pass structure as the main kernel.
    let mut intermediate = [0i32; MAX_INTERMEDIATE_HEIGHT * 4];

    // Horizontal pass (4 columns; `fx` is the one filter, `src_col0 + t` needs no clamp -- the
    // caller proved it in-bounds).
    let ref_base = ref_data.as_ptr();
    for r in 0..intermediate_height {
        let row_ptr = ref_base.add(((src_row0 as usize) + r) * ref_width + src_col0 as usize);
        let mut acc = zero;
        for (t, &tap) in hcoeffs.iter().enumerate() {
            // 4 u16 -> 4 x i32 (zero-extend, matching the scalar `u16 as i32`): exactly the 4
            // columns this tap needs for the 4 outputs, no more (see the Safety bound).
            let vals = _mm_cvtepu16_epi32(_mm_loadl_epi64(row_ptr.add(t) as *const __m128i));
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(vals, _mm_set1_epi32(tap)));
        }
        acc = _mm_srai_epi32(_mm_add_epi32(acc, round_add), 7);
        acc = _mm_min_epi32(_mm_max_epi32(acc, zero), max255);
        _mm_storeu_si128(intermediate.as_mut_ptr().add(r * 4) as *mut __m128i, acc);
    }

    // Vertical pass (4 columns; tap `t` reads intermediate row `r + t` directly, the
    // y_step==16 case).
    for r in 0..h {
        let mut acc = zero;
        for (t, &tap) in vcoeffs.iter().enumerate() {
            let vals = _mm_loadu_si128(intermediate.as_ptr().add((r + t) * 4) as *const __m128i);
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(vals, _mm_set1_epi32(tap)));
        }
        acc = _mm_srai_epi32(_mm_add_epi32(acc, round_add), 7);
        acc = _mm_min_epi32(_mm_max_epi32(acc, zero), max255);
        _mm_storeu_si128(pred.as_mut_ptr().add(r * 4) as *mut __m128i, acc);
    }
}

/// AVX2 mirror of `loop_filter.rs`'s narrow (spec §8.8.5.2, `filter4`), wide 8-tap
/// (§8.8.5.3, `log2_size == 3`) and wide 16-tap (§8.8.5.3, `log2_size == 4`, the `TX_16X16`
/// "wide2" filter) deblocking filters, applied to 8 contiguous along-edge positions on a
/// HORIZONTAL edge (loop_filter.rs pass==1: taps run in the row direction, i.e. `dx=0,dy=1`
/// in that file's terms) -- see docs/implementation-notes.md "SIMD wave 3".
/// Vertical edges (pass==0) stay scalar (along-edge there is consecutive ROWS, strided by the
/// plane stride, not a natural AVX2 fit without a byte gather); `loop_filter.rs`'s dispatch
/// keeps those on the scalar loop.
///
/// `plane_data`/`plane_width` is the raw row-major plane buffer (stride == `plane_width`,
/// `u16`-backed regardless of bit depth -- see `framebuffer.rs`; the caller only dispatches
/// here for `bit_depth == 8`, so every sample is known to fit in `0..=255`).
/// `(x0, y0)` is lane 0's position (loop_filter.rs's `sample_filtering(x, y, ...)` for the
/// first of the 8 lanes); the other 7 lanes are the next 7 contiguous columns
/// `x0+1..=x0+7` at the same row `y0` (this orientation's along-edge axis -- see the
/// `debug_assert_eq!`s at the call site proving contiguity).
///
/// `eligible[lane]` / `is_tx8[lane]` / `is_tx16[lane]` are 0/-1 masks (not `bool`, so the
/// caller can build them once and this fn just loads them): `eligible` gates whether the lane
/// is written at all (false = leave the plane untouched -- inactive lane); `is_tx8` /
/// `is_tx16` are whether the lane's `filter_size` is `TX_8X8` / `TX_16X16` (exactly one of
/// TX_4X4 / TX_8X8 / TX_16X16 per eligible lane; both zero means TX_4X4), mirroring
/// `loop_filter.rs::sample_filtering`'s three-way narrow/wide8/wide16 branch. `limit`/`blimit`/
/// `thresh` are each lane's spec §8.8.4 filter strength (only read for eligible lanes).
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and that reading/writing columns `x0..x0+8` is in
/// bounds for the row window this call touches: rows `y0-4..=y0+3` (the `p3..q3` window spec
/// §8.8.5.1's mask reads) always, AND -- whenever any lane's `is_tx16` is set -- the wider
/// rows `y0-8..=y0+7` (the `p7..q7` window the wide16 filter reads/writes; loaded only in that
/// case). Both windows are exactly what the already-proven-bit-exact scalar
/// `compute_filter_mask`/`wide_filter` touch for an eligible lane at this same edge-constant
/// `y0`, so they are in bounds whenever at least one lane is eligible (and, for the wider
/// window, whenever an `is_tx16` lane is eligible -- the caller only sets `is_tx16` on eligible
/// lanes). Planes are allocated out to superblock boundaries (see `framebuffer::Plane`), so
/// the extra columns read/written for non-eligible lanes in the same 8-wide group are valid
/// memory too (the write blends in the original value there, so it is a no-op).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn loop_filter_horiz8_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    x0: usize,
    y0: usize,
    eligible: &[i32; 8],
    is_tx8: &[i32; 8],
    is_tx16: &[i32; 8],
    limit: &[i32; 8],
    blimit: &[i32; 8],
    thresh: &[i32; 8],
) {
    let base = plane_data.as_mut_ptr();

    // Loads 8 contiguous u16 samples (columns x0..x0+8) from row `y0 + dy`, widened to
    // 8xi32 -- one AVX2 lane per along-edge position, matching `get_off`'s
    // `plane.get(x, y+k)` for pass==1 (`dx=0,dy=1`).
    let load_row = |dy: i64| -> __m256i {
        let row_ptr = base.offset(((y0 as i64 + dy) * plane_width as i64 + x0 as i64) as isize);
        _mm256_cvtepu16_epi32(_mm_loadu_si128(row_ptr as *const __m128i))
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
    let is_tx16_v = _mm256_loadu_si256(is_tx16.as_ptr() as *const __m256i);
    // filter_size >= TX_8X8 (either wide filter is possible): gates flat_mask exactly like the
    // scalar `if filter_size >= TX_8X8`.
    let is_wide_v = _mm256_or_si256(is_tx8_v, is_tx16_v);

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

    // Spec §8.8.5.2 "Narrow filter process" (`filter4`) -- computed unconditionally (cheap,
    // and the common TX_4X4-heavy case selects it; both wide fast paths below reuse it).
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
    // round2(sum-of-8-pixel-values, 3) can't leave that range) down to 8 contiguous u16
    // samples and stores at row `y0 + dy`, columns x0..x0+8 -- the exact reverse of `load_row`.
    let store_row = |dy: i64, v: __m256i| {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let u16x8 = _mm_packus_epi32(lo, hi);
        let row_ptr = base.offset(((y0 as i64 + dy) * plane_width as i64 + x0 as i64) as isize);
        _mm_storeu_si128(row_ptr as *mut __m128i, u16x8);
    };

    // Fast path 1: no lane is a wide filter (every eligible lane is TX_4X4), so flat_mask,
    // wide8 and wide16 can never be selected -- store only the narrow p1..q1 and return. This
    // is the ONLY path that leaves p2/q2 untouched, matching the scalar narrow filter's write
    // set, and it stays inside the narrow p3..q3 window. Common for detailed, TX_4X4-heavy
    // content. `_mm256_testz_si256(a, a)` is a single-instruction "is `a` all-zero" test.
    if _mm256_testz_si256(is_wide_v, is_wide_v) != 0 {
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

    // flat_mask, gated on filter_size >= TX_8X8 (is_wide_v) exactly like the scalar
    // `if filter_size >= TX_8X8 { .. }` (else flat_mask stays false -> narrow selected).
    let one = _mm256_set1_epi32(1);
    let mut fm = gt(abs_diff(p1, p0), one);
    fm = _mm256_or_si256(fm, gt(abs_diff(q1, q0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(p2, p0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(q2, q0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(p3, p0), one));
    fm = _mm256_or_si256(fm, gt(abs_diff(q3, q0), one));
    let flat_mask = _mm256_and_si256(not_mask(fm), is_wide_v);

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

    // Fast path 2: no lane is TX_16X16, so wide16 can't be selected and the p4..p7/q4..q7
    // window is never read -- blend narrow/wide8 into p2..q2 (wide8's write set) and return,
    // exactly as SIMD wave 3 did. Keeps the common case off wide16's 8 extra loads + 14 sums
    // and inside the narrower rows-y0-4..=y0+3 memory window.
    if _mm256_testz_si256(is_tx16_v, is_tx16_v) != 0 {
        let use_wide8 = _mm256_and_si256(elig_and_fm, flat_mask);
        let use_narrow = _mm256_andnot_si256(flat_mask, elig_and_fm);
        // original -> narrow (where selected) -> wide8 (where selected); the two masks are
        // mutually exclusive, so order doesn't matter for overlap.
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
        return;
    }

    // Slow path: at least one eligible lane is TX_16X16. Load the wider p4..p7 / q4..q7 window
    // (rows y0-8..=y0+7). Safe: an eligible TX_16X16 lane's scalar `compute_filter_mask`
    // (flat_mask2) / `wide_filter(4)` reads exactly this window (see the Safety note), so it is
    // in bounds; the group's other columns are valid for the same reason as the p3..q3 window.
    let p4 = load_row(-5);
    let p5 = load_row(-6);
    let p6 = load_row(-7);
    let p7 = load_row(-8);
    let q4 = load_row(4);
    let q5 = load_row(5);
    let q6 = load_row(6);
    let q7 = load_row(7);

    // flat_mask2, gated on filter_size >= TX_16X16 (is_tx16_v) exactly like the scalar
    // `if filter_size >= TX_16X16 { .. }` (else it stays false).
    let mut fm2 = gt(abs_diff(p7, p0), one);
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(q7, q0), one));
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(p6, p0), one));
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(q6, q0), one));
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(p5, p0), one));
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(q5, q0), one));
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(p4, p0), one));
    fm2 = _mm256_or_si256(fm2, gt(abs_diff(q4, q0), one));
    let flat_mask2 = _mm256_and_si256(not_mask(fm2), is_tx16_v);

    // Selection (spec §8.8.5): narrow if !flat_mask; else wide16 if flat_mask2 (which implies
    // TX_16X16); else wide8. The three are mutually exclusive.
    let use_wide16 = _mm256_and_si256(elig_and_fm, _mm256_and_si256(flat_mask, flat_mask2));
    let use_wide8 = _mm256_andnot_si256(flat_mask2, _mm256_and_si256(elig_and_fm, flat_mask));
    let use_narrow = _mm256_andnot_si256(flat_mask, elig_and_fm);

    // Spec §8.8.5.3 "Wide filter process" (log2_size == 4, the 16-tap "wide2" filter). Closed
    // forms derived from the scalar `wide_filter(4)`'s `t = get(i) + sum_j get(clip3(-8,7,i+j))`:
    // each output's 16 weighted terms sum to 16 (round2 by 16), the running sum peaks at
    // 16*255 = 4080 << i32::MAX (no overflow), and integer addition is associative, so this
    // reordering is exact -- verified against the scalar and the sweep in both SIMD configs.
    let round2_4 = |t: __m256i| _mm256_srai_epi32(_mm256_add_epi32(t, _mm256_set1_epi32(8)), 4);
    let sumn = |terms: &[__m256i]| {
        let mut acc = terms[0];
        let mut i = 1;
        while i < terms.len() {
            acc = add(acc, terms[i]);
            i += 1;
        }
        acc
    };
    let op6_wide2 = round2_4(sumn(&[mul(p7, 7), mul(p6, 2), p5, p4, p3, p2, p1, p0, q0]));
    let op5_wide2 = round2_4(sumn(&[
        mul(p7, 6),
        p6,
        mul(p5, 2),
        p4,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
    ]));
    let op4_wide2 = round2_4(sumn(&[
        mul(p7, 5),
        p6,
        p5,
        mul(p4, 2),
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
    ]));
    let op3_wide2 = round2_4(sumn(&[
        mul(p7, 4),
        p6,
        p5,
        p4,
        mul(p3, 2),
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
    ]));
    let op2_wide2 = round2_4(sumn(&[
        mul(p7, 3),
        p6,
        p5,
        p4,
        p3,
        mul(p2, 2),
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        q4,
    ]));
    let op1_wide2 = round2_4(sumn(&[
        mul(p7, 2),
        p6,
        p5,
        p4,
        p3,
        p2,
        mul(p1, 2),
        p0,
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
    ]));
    let op0_wide2 = round2_4(sumn(&[
        p7,
        p6,
        p5,
        p4,
        p3,
        p2,
        p1,
        mul(p0, 2),
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
    ]));
    let oq0_wide2 = round2_4(sumn(&[
        p6,
        p5,
        p4,
        p3,
        p2,
        p1,
        p0,
        mul(q0, 2),
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7,
    ]));
    let oq1_wide2 = round2_4(sumn(&[
        p5,
        p4,
        p3,
        p2,
        p1,
        p0,
        q0,
        mul(q1, 2),
        q2,
        q3,
        q4,
        q5,
        q6,
        mul(q7, 2),
    ]));
    let oq2_wide2 = round2_4(sumn(&[
        p4,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        mul(q2, 2),
        q3,
        q4,
        q5,
        q6,
        mul(q7, 3),
    ]));
    let oq3_wide2 = round2_4(sumn(&[
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        mul(q3, 2),
        q4,
        q5,
        q6,
        mul(q7, 4),
    ]));
    let oq4_wide2 = round2_4(sumn(&[
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        mul(q4, 2),
        q5,
        q6,
        mul(q7, 5),
    ]));
    let oq5_wide2 = round2_4(sumn(&[
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        q4,
        mul(q5, 2),
        q6,
        mul(q7, 6),
    ]));
    let oq6_wide2 = round2_4(sumn(&[p0, q0, q1, q2, q3, q4, q5, mul(q6, 2), mul(q7, 7)]));

    // Blend precedence per position: original -> narrow -> wide8 -> wide16 (the use_* are
    // mutually exclusive, so precedence only decides which candidates a position lists).
    // narrow touches p1..q1; wide8 touches p2..q2; wide16 touches p6..q6 (p7/q7 read-only).
    let blend_w16 = |orig: __m256i, w16: __m256i| _mm256_blendv_epi8(orig, w16, use_wide16);
    let blend_w8_w16 = |orig: __m256i, w8: __m256i, w16: __m256i| {
        _mm256_blendv_epi8(_mm256_blendv_epi8(orig, w8, use_wide8), w16, use_wide16)
    };
    let blend_all = |orig: __m256i, narrow: __m256i, w8: __m256i, w16: __m256i| {
        _mm256_blendv_epi8(
            _mm256_blendv_epi8(_mm256_blendv_epi8(orig, narrow, use_narrow), w8, use_wide8),
            w16,
            use_wide16,
        )
    };

    let p6_out = blend_w16(p6, op6_wide2);
    let p5_out = blend_w16(p5, op5_wide2);
    let p4_out = blend_w16(p4, op4_wide2);
    let p3_out = blend_w16(p3, op3_wide2);
    let p2_out = blend_w8_w16(p2, op2_wide, op2_wide2);
    let p1_out = blend_all(p1, op1_narrow, op1_wide, op1_wide2);
    let p0_out = blend_all(p0, op0_narrow, op0_wide, op0_wide2);
    let q0_out = blend_all(q0, oq0_narrow, oq0_wide, oq0_wide2);
    let q1_out = blend_all(q1, oq1_narrow, oq1_wide, oq1_wide2);
    let q2_out = blend_w8_w16(q2, oq2_wide, oq2_wide2);
    let q3_out = blend_w16(q3, oq3_wide2);
    let q4_out = blend_w16(q4, oq4_wide2);
    let q5_out = blend_w16(q5, oq5_wide2);
    let q6_out = blend_w16(q6, oq6_wide2);

    store_row(-7, p6_out);
    store_row(-6, p5_out);
    store_row(-5, p4_out);
    store_row(-4, p3_out);
    store_row(-3, p2_out);
    store_row(-2, p1_out);
    store_row(-1, p0_out);
    store_row(0, q0_out);
    store_row(1, q1_out);
    store_row(2, q2_out);
    store_row(3, q3_out);
    store_row(4, q4_out);
    store_row(5, q5_out);
    store_row(6, q6_out);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    /// `block_inter_predict_avx2_w4` must produce exactly the leftmost 4 output columns that the
    /// already-sweep-proven 8-wide `block_inter_predict_avx2` produces from the same source and
    /// params: each output column's two-pass FIR depends only on its own source-column
    /// neighborhood, so a width-4 block's 4 columns equal the leftmost 4 of a width-8 block at
    /// the same origin. This pins the width-4 kernel to the width-8 one with no new reference
    /// arithmetic (the sweep is the ultimate gate, but this localizes any width-4 lane/offset
    /// bug immediately).
    #[test]
    fn avx2_w4_matches_w8_leftmost_four_columns() {
        if !avx2_enabled() {
            return;
        }
        let ref_width = 40usize;
        let ref_height = 40usize;
        let mut seed = 0x1234_5678u32;
        let ref_data: Vec<u16> = (0..ref_width * ref_height)
            .map(|_| (xorshift32(&mut seed) & 0xFF) as u16)
            .collect();

        // Interior origin (well away from the buffer edges so every read is in bounds); a
        // couple of block heights, all 4 interp filters, and integer + fractional subpel phases.
        let src_row0 = 8i64;
        let src_col0 = 8i64;
        for &h in &[4usize, 8] {
            let intermediate_height = h + 7;
            for interp_filter in 0..4u8 {
                for &fx in &[0usize, 5, 15] {
                    for &fy in &[0usize, 8, 11] {
                        let mut pred8 = vec![0i32; 8 * h];
                        let mut pred4 = vec![0i32; 4 * h];
                        // SAFETY: avx2_enabled() confirmed; the interior origin keeps every read
                        // within `ref_data`, satisfying both kernels' documented bounds.
                        unsafe {
                            block_inter_predict_avx2(
                                &ref_data,
                                ref_width,
                                src_row0,
                                src_col0,
                                fx,
                                fy,
                                8,
                                h,
                                intermediate_height,
                                interp_filter,
                                &mut pred8,
                            );
                            block_inter_predict_avx2_w4(
                                &ref_data,
                                ref_width,
                                src_row0,
                                src_col0,
                                fx,
                                fy,
                                h,
                                intermediate_height,
                                interp_filter,
                                &mut pred4,
                            );
                        }
                        for r in 0..h {
                            for c in 0..4 {
                                assert_eq!(
                                    pred4[r * 4 + c],
                                    pred8[r * 8 + c],
                                    "h={h} filt={interp_filter} fx={fx} fy={fy} at r={r} c={c}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
