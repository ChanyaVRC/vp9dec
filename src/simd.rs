//! AVX2 SIMD mirror of `predict::block_inter_predict`'s scalar two-pass 8-tap subpel
//! convolution (spec §8.5.2.4): specialized kernels for the common UNSCALED case
//! (`x_step == y_step == 16`, i.e. reference frame same size as the current frame -- see
//! docs/implementation-notes.md "SIMD wave 2") plus a general SCALED-reference kernel
//! (SVC / resize, [`block_inter_predict_scaled_avx2`]). `predict.rs` owns the dispatch and
//! the scalar fallback; this module owns only the vector kernels.
//!
//! x86_64-only for now (`core::arch::x86_64` intrinsics, zero dependencies). A NEON
//! mirror for aarch64 would be a sibling `#[cfg(target_arch = "aarch64")]` module behind
//! the same `predict.rs` dispatch point -- not implemented this wave.

use crate::predict::{MAX_BLOCK_DIM, MAX_INTERMEDIATE_HEIGHT};
use crate::subpel::SUBPEL_FILTERS;
use crate::transform::TxType;
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
/// `intermediate_height <= MAX_INTERMEDIATE_HEIGHT`. Works for all bit depths: `ref_data` is
/// `Plane::as_slice()`'s `u16` buffer (every plane is `u16`-backed, see `framebuffer.rs`), the
/// subpel FIR is bit-depth-agnostic, and the i32 accumulation holds a 12-bit sample through both
/// passes -- only the `clip1` bound differs, so the caller passes `max_val = (1<<bit_depth)-1`
/// and the kernel clips each pass to `0..=max_val`.
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
    max_val: i32,
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
    let max_clip = _mm256_set1_epi32(max_val);

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
            // round2(s, 7) then clip1 (Clip3(0, max_val, .)) -- identical arithmetic and
            // accumulation order to the scalar `round2`/`clip1`, just 8 lanes at once.
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max_clip);
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
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max_clip);
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
    max_val: i32,
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
    let max_clip = _mm_set1_epi32(max_val);

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
        acc = _mm_min_epi32(_mm_max_epi32(acc, zero), max_clip);
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
        acc = _mm_min_epi32(_mm_max_epi32(acc, zero), max_clip);
        _mm_storeu_si128(pred.as_mut_ptr().add(r * 4) as *mut __m128i, acc);
    }
}

/// AVX2 mirror of `predict::block_inter_predict_scalar`'s two-pass 8-tap subpel FIR for the
/// SCALED-reference case (spec §8.5.2.3-8.5.2.4: `x_step`/`y_step` != 16 -- SVC / resize
/// streams, where the reference frame is a different size than the current frame). Same
/// arithmetic, accumulation order, `round2(7)` and `clip1` as the scalar loops -- bit-exact by
/// construction. What differs structurally from the unscaled kernels:
///
/// - **Horizontal pass**: the source position `p = x + x_step*c` gives every output column its
///   own subpel phase `p & 15` (filter row) and source column `p >> 4`. Both are ROW-invariant,
///   so they are precomputed once per call -- per-column gather indices plus the filter taps
///   transposed to tap-major vectors -- and each tap's 8 samples are fetched with
///   `_mm256_i32gather_epi32` from a per-row i32 scratch of the (edge-clamped) source span.
/// - **Vertical pass**: `p = (y & 15) + y_step*r` depends only on the output row, so the phase
///   and source row are uniform across columns -- the same shape as the unscaled vertical pass,
///   just with a per-row filter and base row.
/// - **Edge clamping happens INSIDE the kernel**: the scratch is filled through precomputed
///   `clamp(col, 0, last_x)` source columns and each row through `clamp(row, 0, last_y)`,
///   reproducing the scalar border replication exactly -- so unlike the unscaled kernels there
///   is no `in_bounds` scalar fallback; any block position is valid.
///
/// `w` must be 4 or a multiple of 8; width 4 pads its single 8-lane group by replicating
/// column 3 (the pad lanes' gathers stay in bounds and their results are never stored). Works
/// for all bit depths (the same i32-FIR argument as the unscaled kernels; the caller passes
/// `max_val = (1 << bit_depth) - 1`).
///
/// # Safety
/// The caller must have confirmed `avx2_enabled()`, and must guarantee:
/// `ref_data.len() >= ref_width * ref_height` (every source read is clamped into that
/// rectangle, rows `0..ref_height` x columns `0..ref_width`); `intermediate_height ==
/// (((h-1) * y_step + 15) >> 4) + 8 <= MAX_INTERMEDIATE_HEIGHT`; and the horizontal source
/// span `(((x & 15) + x_step * (w-1)) >> 4) + 8 <= MAX_INTERMEDIATE_HEIGHT`. Both bounds hold
/// for any spec-conformant scaling (§8.5.2.3 caps the steps at 32); the caller re-checks them
/// and falls back to the scalar path so a malformed stream cannot reach here with larger
/// values.
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
    debug_assert!(w == 4 || w.is_multiple_of(8));
    debug_assert!(w <= MAX_BLOCK_DIM && h <= MAX_BLOCK_DIM);
    debug_assert!(intermediate_height <= MAX_INTERMEDIATE_HEIGHT);
    debug_assert_eq!(pred.len(), w * h);
    debug_assert!(ref_data.len() >= ref_width * ref_height);
    debug_assert!(x_step >= 0 && y_step >= 0);

    let filters = &SUBPEL_FILTERS[interp_filter as usize];
    let last_x = ref_width as i64 - 1;
    let last_y = ref_height as i64 - 1;

    // Padded width: w rounded up to a full 8-lane group (only w == 4 pads).
    let wp = if w == 4 { 8 } else { w };
    let groups = wp / 8;

    // Row-invariant per-column precompute (the scalar horizontal loop's `p = x + x_step*c`):
    // gather index (the column's tap-window start `(p >> 4) - 3` relative to `col_min`, i.e.
    // `(p >> 4) - (x >> 4)` -- always >= 0 since x_step >= 0) and the column's 8 filter taps,
    // transposed to tap-major so tap t of a group's 8 columns is one vector. Pad columns
    // (c >= w, only for w == 4) replicate column w-1.
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

    // Clamped source columns for the per-row scratch fill: scratch[i] will hold
    // `ref_row[clamp(col_min + i, 0, last_x)]`, so a gather at index `gidx + t` reads exactly
    // the scalar's `ref_x = clamp((p >> 4) + t - 3, 0, last_x)` sample. Max gather index is
    // `gidx[w-1] + 7 == span - 1`, inside the scratch.
    let mut src_col = [0usize; MAX_INTERMEDIATE_HEIGHT];
    for (i, sc) in src_col[..span].iter_mut().enumerate() {
        *sc = (col_min + i as i64).clamp(0, last_x) as usize;
    }

    let round_add = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let max_clip = _mm256_set1_epi32(max_val);

    // Same fixed-size scratch as the scalar path's `intermediate` (stride `wp`; pad columns
    // hold replicated-column results that the vertical pass computes but never stores).
    let mut intermediate = [0i32; MAX_INTERMEDIATE_HEIGHT * MAX_BLOCK_DIM];
    let mut scratch = [0i32; MAX_INTERMEDIATE_HEIGHT];

    // Horizontal pass: per row, widen the edge-clamped source span to i32 once, then per
    // 8-column group accumulate tap t's gathered samples times its tap-major coefficients --
    // the identical multiply/accumulate/round2(7)/clip1 sequence as the scalar loop.
    for r in 0..intermediate_height {
        let ref_y = ((y >> 4) + r as i64 - 3).clamp(0, last_y) as usize;
        let row = &ref_data[ref_y * ref_width..ref_y * ref_width + ref_width];
        for (dst, &sc) in scratch[..span].iter_mut().zip(src_col[..span].iter()) {
            *dst = row[sc] as i32;
        }
        for (g, (idx_lanes, taps)) in gidx[..groups]
            .iter()
            .zip(hcoef[..groups].iter())
            .enumerate()
        {
            let idx = _mm256_loadu_si256(idx_lanes.as_ptr() as *const __m256i);
            let mut acc = zero;
            for (t, tap_lanes) in taps.iter().enumerate() {
                // The `<4>` gather scale is `size_of::<i32>()` of the `scratch` staging buffer
                // -- a load-bearing coupling: the i32 staging exists precisely because x86 has
                // no u16 gather, so switching `scratch` to u16 would require scale 2 AND a
                // gather instruction that does not exist. The widen-to-i32-then-gather layout
                // is not an optimization choice.
                let vals = _mm256_i32gather_epi32::<4>(scratch.as_ptr().add(t), idx);
                let coeff = _mm256_loadu_si256(tap_lanes.as_ptr() as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vals, coeff));
            }
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max_clip);
            _mm256_storeu_si256(
                intermediate.as_mut_ptr().add(r * wp + g * 8) as *mut __m256i,
                acc,
            );
        }
    }

    // Vertical pass: per output row, one filter (`p & 15`) and one base row (`p >> 4`) shared
    // by every column -- the unscaled kernel's vertical pass with per-row values. `base_row +
    // 7 <= intermediate_height - 1` by the intermediate_height formula (`y & 15 <= 15`).
    for r in 0..h {
        let p = (y & 15) + y_step * (r as i64);
        let vcoeffs = &filters[(p & 15) as usize];
        let base_row = (p >> 4) as usize;
        for g in 0..groups {
            let mut acc = zero;
            for (t, &tap) in vcoeffs.iter().enumerate() {
                let ptr = intermediate.as_ptr().add((base_row + t) * wp + g * 8);
                let vals = _mm256_loadu_si256(ptr as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vals, _mm256_set1_epi32(tap)));
            }
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, round_add), 7);
            acc = _mm256_min_epi32(_mm256_max_epi32(acc, zero), max_clip);
            if w == 4 {
                // The single padded group: store only the 4 real columns.
                _mm_storeu_si128(
                    pred.as_mut_ptr().add(r * 4) as *mut __m128i,
                    _mm256_castsi256_si128(acc),
                );
            } else {
                _mm256_storeu_si256(pred.as_mut_ptr().add(r * w + g * 8) as *mut __m256i, acc);
            }
        }
    }
}

/// AVX2 mirror of `loop_filter.rs`'s narrow (spec §8.8.5.2, `filter4`), wide 8-tap
/// (§8.8.5.3, `log2_size == 3`) and wide 16-tap (§8.8.5.3, `log2_size == 4`, the `TX_16X16`
/// "wide2" filter) deblocking filters, applied to 8 contiguous along-edge positions on a
/// HORIZONTAL edge (loop_filter.rs pass==1: taps run in the row direction, i.e. `dx=0,dy=1`
/// in that file's terms) -- see docs/implementation-notes.md "SIMD wave 3".
/// Vertical edges (pass==0) are handled by [`loop_filter_vert8_avx2`], which transposes the tap
/// window into this kernel's row-major layout and reuses this exact arithmetic.
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
    bit_depth: u8,
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
    // The 8-bit base (128) and clamp range (-128..=127) scale by `<< (bit_depth - 8)`, matching
    // `loop_filter.rs::narrow_filter`'s `half` / `clamp_hi` / `clamp_lo` (identity at 8-bit).
    let clamp_hi = (128i32 << (bit_depth - 8)) - 1;
    let clamp_lo = -(clamp_hi + 1);
    let c128 = _mm256_set1_epi32(1 << (bit_depth - 1));
    let clamp4 = |v: __m256i| {
        _mm256_min_epi32(
            _mm256_max_epi32(v, _mm256_set1_epi32(clamp_lo)),
            _mm256_set1_epi32(clamp_hi),
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
    // `if filter_size >= TX_8X8 { .. }` (else flat_mask stays false -> narrow selected). The
    // flat threshold is 1 at 8-bit and scales `<< (bit_depth - 8)` (scalar's `threshold`).
    let one = _mm256_set1_epi32(1 << (bit_depth - 8));
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

/// Transposes an 8x8 matrix of `u16` (`r[i]` = input row `i`, 8 lanes) so the returned
/// `out[j]` holds input column `j` across all 8 rows. The standard SSE2 unpack network
/// (interleave by 16-, then 32-, then 64-bit granularity); it is its own inverse. Only SSE2
/// (baseline on x86_64), so callable from any x86_64 context. Used by [`loop_filter_vert8_avx2`]
/// to turn a vertical edge (taps along a row) into the row-major layout the horizontal kernel
/// expects, and to turn the filtered result back.
#[inline]
unsafe fn transpose8x8_u16(r: &[__m128i; 8]) -> [__m128i; 8] {
    let a0 = _mm_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm_unpackhi_epi16(r[0], r[1]);
    let a2 = _mm_unpacklo_epi16(r[2], r[3]);
    let a3 = _mm_unpackhi_epi16(r[2], r[3]);
    let a4 = _mm_unpacklo_epi16(r[4], r[5]);
    let a5 = _mm_unpackhi_epi16(r[4], r[5]);
    let a6 = _mm_unpacklo_epi16(r[6], r[7]);
    let a7 = _mm_unpackhi_epi16(r[6], r[7]);
    let b0 = _mm_unpacklo_epi32(a0, a2);
    let b1 = _mm_unpackhi_epi32(a0, a2);
    let b2 = _mm_unpacklo_epi32(a1, a3);
    let b3 = _mm_unpackhi_epi32(a1, a3);
    let b4 = _mm_unpacklo_epi32(a4, a6);
    let b5 = _mm_unpackhi_epi32(a4, a6);
    let b6 = _mm_unpacklo_epi32(a5, a7);
    let b7 = _mm_unpackhi_epi32(a5, a7);
    [
        _mm_unpacklo_epi64(b0, b4),
        _mm_unpackhi_epi64(b0, b4),
        _mm_unpacklo_epi64(b1, b5),
        _mm_unpackhi_epi64(b1, b5),
        _mm_unpacklo_epi64(b2, b6),
        _mm_unpackhi_epi64(b2, b6),
        _mm_unpacklo_epi64(b3, b7),
        _mm_unpackhi_epi64(b3, b7),
    ]
}

/// AVX2 VERTICAL-edge deblocking filter (`loop_filter.rs` pass==0: taps run along a row, i.e.
/// `dx=1,dy=0`), the transpose of [`loop_filter_horiz8_avx2`]. The 8 along-edge positions here
/// are 8 consecutive ROWS `y0..y0+7` at column `x0` (the along-edge axis is strided by the
/// plane, the taps `x0-8..x0+7` are contiguous within each row) -- so this transposes the tap
/// window into the row-major layout the horizontal kernel wants, runs that already-proven
/// kernel unchanged, and transposes the result back. All the bit-exact filter arithmetic (mask,
/// narrow, wide8, wide16) is thus shared verbatim; only the load/store orientation differs.
///
/// `plane_data`/`plane_width`, the 0/-1 lane masks, and `limit`/`blimit`/`thresh` mean exactly
/// what they do for [`loop_filter_horiz8_avx2`]; `(x0, y0)` is lane 0's position and the other 7
/// lanes are the next 7 rows `y0+1..=y0+7` at the same column `x0`.
///
/// # Safety
/// Caller must confirm `avx2_enabled()` and that the tap window is in bounds: columns
/// `x0-4..=x0+3` across rows `y0..=y0+7` always, AND -- whenever any lane's `is_tx16` is set --
/// the wider columns `x0-8..=x0+7`. This is the transpose of the horizontal kernel's contract
/// (rows<->cols swapped) and holds under the same reasoning: planes are allocated out to
/// superblock boundaries in both dimensions (see `framebuffer::Plane`), the wider window is only
/// read when an eligible `is_tx16` lane forces a 16-aligned interior edge, and non-eligible
/// lanes/columns are written back with their original transposed-in value (a no-op).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn loop_filter_vert8_avx2(
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
    bit_depth: u8,
) {
    // Wide window (16 tap columns x0-8..x0+7) iff a lane needs the TX_16X16 wide16 filter, else
    // the narrow window (8 tap columns x0-4..x0+3). Mirrors the horizontal kernel's own is_tx16
    // branch and its memory-window contract: the wider columns are only read when an is_tx16
    // lane is present, which the caller sets only for eligible TX_16X16 (16-aligned) lanes.
    let wide = is_tx16.iter().any(|&t| t != 0);
    let center = if wide { 8usize } else { 4usize }; // q0's tap-row in the scratch
    let col0 = x0 - center; // leftmost tap column (p3 for the narrow window, p7 for the wide)

    let base = plane_data.as_mut_ptr();
    // scratch: up to 16 tap-rows x 8 lane-cols, row-major stride 8 -- the layout the horizontal
    // kernel reads (tap axis = scratch rows, along-edge axis = scratch cols). Only the first
    // `_win` rows are filled/used; the narrow case leaves rows 8..15 untouched (never read).
    let mut scratch = [0u16; 16 * 8];

    // Transpose the plane tap-window (8 rows y0..y0+7 x `_win` cols from col0) into scratch, one
    // 8x8 block per 8 columns. Afterwards scratch row `tr` is tap column `col0+tr` across the 8
    // lanes -- so scratch row `center` is q0's column x0, matching a horizontal edge at row
    // `center`.
    let mut rows = [_mm_setzero_si128(); 8];
    for (r, slot) in rows.iter_mut().enumerate() {
        *slot = _mm_loadu_si128(base.add((y0 + r) * plane_width + col0) as *const __m128i);
    }
    let t = transpose8x8_u16(&rows);
    for (tr, &v) in t.iter().enumerate() {
        _mm_storeu_si128(scratch.as_mut_ptr().add(tr * 8) as *mut __m128i, v);
    }
    if wide {
        for (r, slot) in rows.iter_mut().enumerate() {
            *slot = _mm_loadu_si128(base.add((y0 + r) * plane_width + col0 + 8) as *const __m128i);
        }
        let t = transpose8x8_u16(&rows);
        for (tr, &v) in t.iter().enumerate() {
            _mm_storeu_si128(scratch.as_mut_ptr().add((8 + tr) * 8) as *mut __m128i, v);
        }
    }

    // Apply the proven horizontal kernel to the scratch: a width-8 "plane" whose edge is at row
    // `center` (q0). Its own narrow/wide8/wide16 selection and p3..q3 / p7..q7 window reads land
    // exactly on the scratch rows filled above.
    loop_filter_horiz8_avx2(
        &mut scratch,
        8,
        0,
        center,
        eligible,
        is_tx8,
        is_tx16,
        limit,
        blimit,
        thresh,
        bit_depth,
    );

    // Transpose the (now filtered) scratch back to the plane. Rows the kernel left unwritten keep
    // the original values transposed in above, so writing the whole window back is a no-op there.
    for (tr, slot) in rows.iter_mut().enumerate() {
        *slot = _mm_loadu_si128(scratch.as_ptr().add(tr * 8) as *const __m128i);
    }
    let out = transpose8x8_u16(&rows);
    for (r, &v) in out.iter().enumerate() {
        _mm_storeu_si128(base.add((y0 + r) * plane_width + col0) as *mut __m128i, v);
    }
    if wide {
        for (tr, slot) in rows.iter_mut().enumerate() {
            *slot = _mm_loadu_si128(scratch.as_ptr().add((8 + tr) * 8) as *const __m128i);
        }
        let out = transpose8x8_u16(&rows);
        for (r, &v) in out.iter().enumerate() {
            _mm_storeu_si128(
                base.add((y0 + r) * plane_width + col0 + 8) as *mut __m128i,
                v,
            );
        }
    }
}

// ===========================================================================================
// SIMD inverse DCT (SIMD wave 4b; 10/12-bit added later). The scalar `transform::idct` is a
// recursive butterfly network on an i64 array; the functions below mirror it VERBATIM on i32
// vectors, each element vector holding that transform element across 8 rows (8 lanes) -- so 8
// independent 1D row/column IDCTs run in parallel. Lane STORAGE is i32 at every bit depth:
// spec §8.7.1.1 bounds every value stored into T to signed `8 + BitDepth` bits (<= 20 at
// 12-bit), so loads/stores, transposes, `h_op` sums and the final `round2` all fit i32. Only
// the `t * cos64` PRODUCTS inside `b_op` differ by depth: at bit_depth == 8 they fit i32
// (16b * 15b), so `_mm256_mullo_epi32` is exact (`b_op_simd`); at 10/12-bit they reach ~2^33
// and would wrap, so the HBD network (`HBD = true`) computes them with 32x32 -> 64-bit
// widening multiplies and rounds in i64 (`b_op_simd_hbd`). DCT_DCT (both axes DCT) is
// vectorized at every depth; the ADST-containing types (ADST_DCT / DCT_ADST / ADST_ADST, sizes
// 4/8/16 -- 32x32 is DCT-only) at 8-bit only (see the inverse-ADST section below). WHT
// (lossless) and 10/12-bit ADST stay scalar.
// ===========================================================================================

/// `round2(x, 14)` (`transform.rs::round2`) lane-wise; the b_op products fit i32 (see the module
/// comment above), so `(x + 2^13) >> 14` arithmetic-shifted matches the scalar i64 round2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round2_14(x: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_add_epi32(x, _mm256_set1_epi32(1 << 13)), 14)
}

/// Vector `B(a, b, angle, flip)` butterfly (`transform.rs::b_op`) on 8 rows at once.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn b_op_simd(t: &mut [__m256i], a: usize, b: usize, angle: i32, flip: bool) {
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
unsafe fn b_op_simd_hbd(t: &mut [__m256i], a: usize, b: usize, angle: i32, flip: bool) {
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
unsafe fn h_op_simd(t: &mut [__m256i], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = _mm256_add_epi32(x, y);
    t[b] = _mm256_sub_epi32(x, y);
}

/// Vector `idct_permute` (`transform.rs::idct_permute`): bit-reversal reorder of the element
/// vectors (no arithmetic, so trivially the same at i32).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn idct_permute_simd(t: &mut [__m256i], n: u32) {
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
unsafe fn idct_simd<const HBD: bool>(t: &mut [__m256i], n: u32) {
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

// -------------------------------------------------------------------------------------------
// SIMD inverse ADST, 8-bit ONLY. The scalar `transform::iadst4/8/16` networks are mirrored
// verbatim on i32 8-lane element vectors, exactly like the DCT above. Overflow safety at
// bit_depth == 8 (from spec §8.7.1.1 / §8.7.2 conformance: every stored T value fits signed
// `8 + BitDepth == 16` bits, so |T| <= 2^15; |cos64| <= 2^14; max |cos64|+|sin64| == 23170):
//   - every `SB` product `T * cos64` is <= 2^29, so `_mm256_mullo_epi32` is exact;
//   - every unrounded `S` value is <= 2^15 * 23170 < 2^30, inside the i32 lanes;
//   - every `SH` sum `S[a] +/- S[b]` (+ the 2^13 round) is < 2^31;
//   - the iadst4 chains peak at (SINPI_1+SINPI_2+SINPI_3+SINPI_4) * 2^15 = 43801 * 2^15 < 2^31
//     (`SINPI_1_9 + SINPI_2_9 == SINPI_4_9` exactly, which bounds the `x0 + x1` intermediate),
//     and its largest product `SINPI_3_9 * (T0 - T2 + T3)` is <= 13377 * 3 * 2^15 < 2^31;
//   - the networks' `B`/`H` ops are bounded as in the DCT case (`b_op_simd`'s own claim).
// At 10/12-bit the spec's bound on the `S` array is `24 + BitDepth` (up to 36) bits -- beyond
// i32 LANE STORAGE, not merely the products -- so the DCT's products-only i64 widening does not
// carry over; 10/12-bit ADST stays scalar and the dispatch (`tile/residual.rs`) gates these
// kernels on bit_depth == 8.
// -------------------------------------------------------------------------------------------

/// Vector `SB(a, b, angle, flip)` (`transform.rs::sb_op`, spec §8.7.1.1) on 8 rows: the
/// butterfly rotation into the high-precision `S` array, unrounded. 8-bit only -- see the
/// section comment for why every product and `S` value fits i32 there.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sb_op_simd(s: &mut [__m256i], t: &[__m256i], a: usize, b: usize, angle: i32, flip: bool) {
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
/// from `S` back into `T`. The `S[a] +/- S[b]` sums fit i32 at 8-bit (see the section comment).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sh_op_simd(t: &mut [__m256i], s: &[__m256i], a: usize, b: usize) {
    t[a] = round2_14(_mm256_add_epi32(s[a], s[b]));
    t[b] = round2_14(_mm256_sub_epi32(s[a], s[b]));
}

/// Vector `adst_input_permute` (`transform.rs`, spec §8.7.1.4): pure reorder, no arithmetic.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn adst_input_permute_simd(t: &mut [__m256i], n: u32) {
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
/// (`n` is 3 or 4).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn adst_output_permute_simd(t: &mut [__m256i], n: u32) {
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

/// Vector inverse ADST4 (`transform.rs::iadst4_impl`, spec §8.7.1.6) on 8 rows. Same statement
/// order and evaluation order as the scalar (`v` is `(T0 - T2) + T3`, `o3` is `(x0 + x1) - x3`).
/// i32 is exact here because NO intermediate wraps at all: the section comment's chain bounds
/// keep every product, partial sum, and round2 input below 2^31 in magnitude at 8-bit. (A
/// weaker "wrapped adds cancel if the final value fits" argument would NOT suffice -- the
/// chains end in `round2` shifts, and a shift of a wrapped sum is not the shift of the true
/// sum -- so keep the no-wrap bound intact when touching this network.)
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

/// Vector inverse ADST8 (`transform.rs::iadst8_impl`, spec §8.7.1.7) on 8 rows -- a verbatim
/// mirror of the scalar op sequence.
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

/// Vector inverse ADST16 (`transform.rs::iadst16_impl`, spec §8.7.1.8) on 8 rows -- a verbatim
/// mirror of the scalar op sequence.
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

/// Vector `iadst(t, n)` (`transform.rs::iadst`, spec §8.7.1.9): ADST4/8/16 by size. The input/
/// output permutes are inside the size-specific bodies, mirroring the scalar structure.
#[target_feature(enable = "avx2")]
unsafe fn iadst_simd(t: &mut [__m256i], n: u32) {
    match n {
        2 => iadst4_simd(t),
        3 => iadst8_simd(t),
        4 => iadst16_simd(t),
        _ => unreachable!("iadst_simd only supports n = 2..=4"),
    }
}

/// Transposes a 4x4 i32 matrix (`r[i]` = row i) so `out[j]` = column j.
#[inline]
unsafe fn transpose4x4_i32(r: &[__m128i; 4]) -> [__m128i; 4] {
    let a0 = _mm_unpacklo_epi32(r[0], r[1]);
    let a1 = _mm_unpackhi_epi32(r[0], r[1]);
    let a2 = _mm_unpacklo_epi32(r[2], r[3]);
    let a3 = _mm_unpackhi_epi32(r[2], r[3]);
    [
        _mm_unpacklo_epi64(a0, a2),
        _mm_unpackhi_epi64(a0, a2),
        _mm_unpacklo_epi64(a1, a3),
        _mm_unpackhi_epi64(a1, a3),
    ]
}

/// Transposes an 8x8 i32 matrix (`r[i]` = row i) so `out[j]` = column j. Unpack network within
/// each 128-bit lane, then a `permute2x128` pass to swap the diagonal 128-bit blocks.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose8x8_i32(r: &[__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let s0 = _mm256_unpacklo_epi64(t0, t2);
    let s1 = _mm256_unpackhi_epi64(t0, t2);
    let s2 = _mm256_unpacklo_epi64(t1, t3);
    let s3 = _mm256_unpackhi_epi64(t1, t3);
    let s4 = _mm256_unpacklo_epi64(t4, t6);
    let s5 = _mm256_unpackhi_epi64(t4, t6);
    let s6 = _mm256_unpacklo_epi64(t5, t7);
    let s7 = _mm256_unpackhi_epi64(t5, t7);
    [
        _mm256_permute2x128_si256(s0, s4, 0x20),
        _mm256_permute2x128_si256(s1, s5, 0x20),
        _mm256_permute2x128_si256(s2, s6, 0x20),
        _mm256_permute2x128_si256(s3, s7, 0x20),
        _mm256_permute2x128_si256(s0, s4, 0x31),
        _mm256_permute2x128_si256(s1, s5, 0x31),
        _mm256_permute2x128_si256(s2, s6, 0x31),
        _mm256_permute2x128_si256(s3, s7, 0x31),
    ]
}

/// `round2(x, shift)` with a runtime shift (the 2D column pass's `min(n+2, 6)`), lane-wise.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round2_var(x: __m256i, shift: u32) -> __m256i {
    let add = _mm256_set1_epi32(1 << (shift - 1));
    _mm256_sra_epi32(_mm256_add_epi32(x, add), _mm_cvtsi32_si128(shift as i32))
}

/// One separable pass of the 2D inverse transform: transforms every "row" of `input` (n0 x n0,
/// row-major i32) with the inverse DCT (`ADST == false`) or inverse ADST (`ADST == true`) and
/// writes the result TRANSPOSED into `output`, so running this twice (row then column) lands
/// the 2D result back in row-major. `round_shift` is `None` for the row pass and
/// `Some(min(n+2,6))` for the column pass (spec §8.7.2). Vectorizes across 8 rows at a time
/// (4 for n0==4); each element vector holds one transform coefficient across those rows.
/// The ADST variant is 8-bit-only i32 arithmetic (see the inverse-ADST section comment) and is
/// never instantiated together with `HBD`.
#[target_feature(enable = "avx2")]
unsafe fn xform_pass<const HBD: bool, const ADST: bool>(
    input: &[i32; 1024],
    output: &mut [i32; 1024],
    n: u32,
    round_shift: Option<u32>,
) {
    let n0 = 1usize << n;

    if n0 == 4 {
        let mut rows4 = [_mm_setzero_si128(); 4];
        for (r, slot) in rows4.iter_mut().enumerate() {
            *slot = _mm_loadu_si128(input.as_ptr().add(r * 4) as *const __m128i);
        }
        let cols = transpose4x4_i32(&rows4);
        let mut t = [_mm256_setzero_si256(); 32];
        for (c, &col) in cols.iter().enumerate() {
            // The upper 128 bits of `_mm256_castsi128_si256` are formally UNDEFINED. Safe here
            // only because everything downstream on this n0 == 4 path is lane-wise (butterfly
            // add/sub/mul/shift; the ADST/DCT permutes reorder whole element VECTORS, not
            // lanes) and the final store reads only the low 128 bits. Adding any lane-CROSSING
            // op (cross-half permute/shuffle, horizontal reduce) here requires zero-extending
            // instead (`_mm256_zextsi128_si256`).
            t[c] = _mm256_castsi128_si256(col);
        }
        if ADST {
            iadst_simd(&mut t, n);
        } else {
            idct_permute_simd(&mut t, n);
            idct_simd::<HBD>(&mut t, n);
        }
        for (k, tk) in t[..4].iter().enumerate() {
            let v = round_shift.map_or(*tk, |sh| round2_var(*tk, sh));
            _mm_storeu_si128(
                output.as_mut_ptr().add(k * 4) as *mut __m128i,
                _mm256_castsi256_si128(v),
            );
        }
        return;
    }

    let mut cs = 0usize;
    while cs < n0 {
        // Build element vectors t[0..n0] (t[k] = coefficient k across rows cs..cs+8) by
        // transposing the 8-row chunk one 8-column block at a time.
        let mut t = [_mm256_setzero_si256(); 32];
        let mut g = 0usize;
        while g < n0 {
            let mut rows8 = [_mm256_setzero_si256(); 8];
            for (r, slot) in rows8.iter_mut().enumerate() {
                *slot = _mm256_loadu_si256(input.as_ptr().add((cs + r) * n0 + g) as *const __m256i);
            }
            let cols = transpose8x8_i32(&rows8);
            t[g..g + 8].copy_from_slice(&cols);
            g += 8;
        }
        if ADST {
            iadst_simd(&mut t, n);
        } else {
            idct_permute_simd(&mut t, n);
            idct_simd::<HBD>(&mut t, n);
        }
        for (k, tk) in t[..n0].iter().enumerate() {
            let v = round_shift.map_or(*tk, |sh| round2_var(*tk, sh));
            _mm256_storeu_si256(output.as_mut_ptr().add(k * n0 + cs) as *mut __m256i, v);
        }
        cs += 8;
    }
}

/// The two separable passes of the AVX2 DCT_DCT 2D inverse transform (spec §8.7.2):
/// converts `dequant` (n0 x n0 row-major i64) to i32, runs the row pass (-> R^T) then the column
/// pass (-> O, with `round2(min(n+2,6))`), and writes O row-major into `out`. Bit-exact with the
/// scalar path for conformant streams (see the SIMD-inverse-DCT module comment): spec §8.7.1.1
/// keeps every stored value inside i32 at any depth, so the narrower lane arithmetic is
/// identical -- provided `HBD` matches the content (widened butterfly products for 10/12-bit).
/// `dequant.len()` must be `n0*n0` (`2 <= n <= 5`).
#[target_feature(enable = "avx2")]
unsafe fn idct_2d_dct<const HBD: bool>(dequant: &[i64], n: u32, out: &mut [i32; 1024]) {
    let count = (1usize << n) * (1usize << n);
    let mut buf = [0i32; 1024];
    let mut buf2 = [0i32; 1024];
    for (dst, &src) in buf[..count].iter_mut().zip(dequant[..count].iter()) {
        *dst = src as i32;
    }
    xform_pass::<HBD, false>(&buf, &mut buf2, n, None); // row pass -> buf2 == R^T
    xform_pass::<HBD, false>(&buf2, out, n, Some((n + 2).min(6))); // column pass -> out == O
}

/// AVX2 8-bit DCT_DCT inverse transform + reconstruction (SIMD wave 4b), fused: transforms
/// `dequant` and adds the residual straight into the plane with the 8-bit clip
/// (`clip(pred + residual, 0, 255)`), skipping both the i64 write-back a standalone transform
/// would do and the scalar per-pixel reconstruction loop. The n0 x n0 block sits at
/// `(start_x, start_y)` in the row-major u16 `plane_data` (stride `plane_width`). Bit-exact with
/// the scalar transform-then-reconstruct at `bit_depth == 8`.
///
/// # Safety
/// `avx2_enabled()` must hold; `dequant.len()` must be `n0*n0` (`2 <= n <= 5`); and the block's
/// rows `start_y..start_y+n0` x columns `start_x..start_x+n0` must be in bounds for `plane_data`.
#[target_feature(enable = "avx2")]
pub unsafe fn inverse_transform_dct_dct_reconstruct_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    dequant: &[i64],
    n: u32,
) {
    idct_dct_reconstruct::<false>(plane_data, plane_width, start_x, start_y, dequant, n, 255);
}

/// High-bit-depth (10/12-bit) companion to [`inverse_transform_dct_dct_reconstruct_avx2`]: the
/// same fused DCT_DCT inverse transform + reconstruction, but with the butterfly products
/// widened to i64 (see [`b_op_simd_hbd`] -- at 10/12-bit they overflow the 8-bit kernel's i32
/// `mullo`) and the reconstruction clip at `(1 << bit_depth) - 1`. Bit-exact with the scalar
/// transform-then-reconstruct for conformant 10/12-bit streams (spec §8.7.1.1's `8 + BitDepth`
/// bound keeps all stored intermediates inside the i32 lanes).
///
/// # Safety
/// Same contract as [`inverse_transform_dct_dct_reconstruct_avx2`]; `bit_depth` must be the
/// stream's bit depth (10 or 12).
#[target_feature(enable = "avx2")]
pub unsafe fn inverse_transform_dct_dct_reconstruct_hbd_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    dequant: &[i64],
    n: u32,
    bit_depth: u8,
) {
    let max_val = (1i32 << bit_depth) - 1;
    idct_dct_reconstruct::<true>(
        plane_data,
        plane_width,
        start_x,
        start_y,
        dequant,
        n,
        max_val,
    );
}

/// AVX2 8-bit inverse transform + reconstruction for the ADST-containing transform types
/// (`AdstDct` / `DctAdst` / `AdstAdst`, spec §8.7.1.4-8.7.1.9 + §8.7.2), fused into the plane
/// exactly like [`inverse_transform_dct_dct_reconstruct_avx2`]: the ADST axis runs the vector
/// `iadst_simd`, the DCT axis (of the mixed types) the proven `idct_simd`, and the residual is
/// added with the 8-bit clip. ADST exists only for 4x4/8x8/16x16 (`n <= 4`; 32x32 is DCT-only,
/// spec §8.7.2). 8-bit only -- see the inverse-ADST section comment; the dispatch keeps
/// 10/12-bit ADST and lossless WHT scalar.
///
/// # Safety
/// Same contract as [`inverse_transform_dct_dct_reconstruct_avx2`], with `2 <= n <= 4`;
/// `tx_type` must not be `DctDct` (that type has its own fused entries) and the stream must be
/// 8-bit.
#[target_feature(enable = "avx2")]
pub unsafe fn inverse_transform_adst_reconstruct_avx2(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    dequant: &[i64],
    n: u32,
    tx_type: TxType,
) {
    debug_assert!((2..=4).contains(&n));
    debug_assert!(tx_type != TxType::DctDct);
    let n0 = 1usize << n;
    let count = n0 * n0;
    let mut buf = [0i32; 1024];
    let mut buf2 = [0i32; 1024];
    let mut o = [0i32; 1024];
    for (dst, &src) in buf[..count].iter_mut().zip(dequant[..count].iter()) {
        *dst = src as i32;
    }
    let shift = (n + 2).min(6);
    // Row pass -> buf2 == R^T. The scalar 2D driver (`transform.rs::inverse_transform_block`)
    // runs ADST rows for DctAdst | AdstAdst and DCT rows otherwise; columns are the complement.
    match tx_type {
        TxType::DctAdst | TxType::AdstAdst => xform_pass::<false, true>(&buf, &mut buf2, n, None),
        _ => xform_pass::<false, false>(&buf, &mut buf2, n, None),
    }
    // Column pass -> o == O (row-major), with the final `round2(min(n+2, 6))`.
    match tx_type {
        TxType::AdstDct | TxType::AdstAdst => {
            xform_pass::<false, true>(&buf2, &mut o, n, Some(shift))
        }
        _ => xform_pass::<false, false>(&buf2, &mut o, n, Some(shift)),
    }
    reconstruct_add_clip(plane_data, plane_width, start_x, start_y, &o, n0, 255);
}

/// Shared body of the two fused DCT_DCT entries above: 2D transform (`idct_2d_dct::<HBD>`),
/// then the fused per-pixel reconstruction (`reconstruct_add_clip`).
#[target_feature(enable = "avx2")]
unsafe fn idct_dct_reconstruct<const HBD: bool>(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    dequant: &[i64],
    n: u32,
    max_val: i32,
) {
    let n0 = 1usize << n;
    let mut o = [0i32; 1024];
    idct_2d_dct::<HBD>(dequant, n, &mut o);
    reconstruct_add_clip(plane_data, plane_width, start_x, start_y, &o, n0, max_val);
}

/// Per-pixel `clip(pred + residual, 0, max_val)` of the n0 x n0 residual block `o` (row-major)
/// straight into the u16 plane -- the fused reconstruction shared by the DCT_DCT and ADST
/// entries.
#[target_feature(enable = "avx2")]
unsafe fn reconstruct_add_clip(
    plane_data: &mut [u16],
    plane_width: usize,
    start_x: usize,
    start_y: usize,
    o: &[i32; 1024],
    n0: usize,
    max_val: i32,
) {
    let base = plane_data.as_mut_ptr();
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi32(max_val);

    if n0 == 4 {
        for i in 0..4 {
            let row = base.add((start_y + i) * plane_width + start_x);
            let pred = _mm_cvtepu16_epi32(_mm_loadl_epi64(row as *const __m128i));
            let resid = _mm_loadu_si128(o.as_ptr().add(i * 4) as *const __m128i);
            let sum = _mm_add_epi32(pred, resid);
            let clipped = _mm_min_epi32(
                _mm_max_epi32(sum, _mm256_castsi256_si128(zero)),
                _mm256_castsi256_si128(max),
            );
            _mm_storel_epi64(row as *mut __m128i, _mm_packus_epi32(clipped, clipped));
        }
        return;
    }

    for i in 0..n0 {
        let mut j = 0usize;
        while j < n0 {
            let row = base.add((start_y + i) * plane_width + start_x + j);
            let pred = _mm256_cvtepu16_epi32(_mm_loadu_si128(row as *const __m128i));
            let resid = _mm256_loadu_si256(o.as_ptr().add(i * n0 + j) as *const __m256i);
            let sum = _mm256_add_epi32(pred, resid);
            let clipped = _mm256_min_epi32(_mm256_max_epi32(sum, zero), max);
            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(clipped),
                _mm256_extracti128_si256(clipped, 1),
            );
            _mm_storeu_si128(row as *mut __m128i, packed);
            j += 8;
        }
    }
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
                                255,
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
                                255,
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

    /// The scaled-reference kernel (`block_inter_predict_scaled_avx2`) must exactly reproduce
    /// the scalar `predict::block_inter_predict_scalar` across: widths (4 -- the padded single
    /// group -- and the 8-wide-group multiples), scale steps on both axes (upscale 1, the
    /// spec-conformant range up to the §8.5.2.3 maximum 32, mixed x/y including one axis
    /// unscaled), start positions engaging the left/top edge clamp, the right/bottom clamp,
    /// and interior subpel phases, and all bit depths (8/10/12 via `max_val`). The filter
    /// index rotates pseudo-randomly across the matrix, covering all 4 filter banks over the
    /// run. The (w=64, x_step=32) case lands `span` exactly on the MAX_INTERMEDIATE_HEIGHT
    /// scratch bound.
    #[test]
    fn scaled_inter_predict_avx2_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let ref_w = 48usize;
        let ref_h = 40usize;
        let mut seed = 0x5CA1_ED01u32;
        for &bit_depth in &[8u8, 10, 12] {
            let sample_mask = (1u32 << bit_depth) - 1;
            let mut plane = crate::framebuffer::Plane::new(ref_w, ref_h);
            for yy in 0..ref_h {
                for xx in 0..ref_w {
                    plane.set(xx, yy, (xorshift32(&mut seed) & sample_mask) as u16);
                }
            }
            for &(w, h) in &[(4usize, 4usize), (4, 8), (8, 8), (16, 4), (32, 8), (64, 4)] {
                for &(x_step, y_step) in &[
                    (1i64, 32i64),
                    (8, 8),
                    (9, 27),
                    (16, 24),
                    (24, 16),
                    (31, 31),
                    (32, 32),
                ] {
                    for &(x, y) in &[
                        (-97i64, -70i64),                           // heavy left + top clamp
                        (3, 7),                                     // interior, small phases
                        (37 * 16 + 11, 21 * 16 + 5),                // interior / right-bottom clamp
                        ((ref_w as i64) << 4, (ref_h as i64) << 4), // fully past right/bottom
                    ] {
                        let interp_filter = (xorshift32(&mut seed) & 3) as u8;
                        let intermediate_height =
                            ((((h as i64 - 1) * y_step + 15) >> 4) + 8) as usize;
                        let span = ((((x & 15) + x_step * (w as i64 - 1)) >> 4) + 8) as usize;
                        assert!(
                            intermediate_height <= MAX_INTERMEDIATE_HEIGHT
                                && span <= MAX_INTERMEDIATE_HEIGHT
                        );
                        let mut pred_scalar = vec![0i32; w * h];
                        crate::predict::block_inter_predict_scalar(
                            &plane,
                            x,
                            y,
                            x_step,
                            y_step,
                            w,
                            h,
                            interp_filter,
                            bit_depth,
                            &mut pred_scalar,
                        );
                        let mut pred_simd = vec![0i32; w * h];
                        // SAFETY: avx2_enabled() confirmed; the plane slice is exactly
                        // ref_w * ref_h and both scratch bounds were just asserted, matching
                        // the kernel's documented contract (it clamps all source reads).
                        unsafe {
                            block_inter_predict_scaled_avx2(
                                plane.as_slice(),
                                ref_w,
                                ref_h,
                                x,
                                y,
                                x_step,
                                y_step,
                                w,
                                h,
                                intermediate_height,
                                interp_filter,
                                (1i32 << bit_depth) - 1,
                                &mut pred_simd,
                            );
                        }
                        assert_eq!(
                            pred_scalar, pred_simd,
                            "bd={bit_depth} w={w} h={h} steps=({x_step},{y_step}) \
                             pos=({x},{y}) filt={interp_filter}"
                        );
                    }
                }
            }
        }
    }

    /// The vector 1D inverse DCT (`idct_permute_simd` + `idct_simd`) must exactly reproduce the
    /// scalar `transform::idct_permute` + `transform::idct` on each of its 8 lanes, for every
    /// size n = 2..=5. Inputs are kept small (structure-validating) so no intermediate exceeds
    /// i32 -- magnitude-independent index/angle/flip bugs still show up, and the i32-vs-i64
    /// agreement on realistic ranges is covered end-to-end by the 2D test and the official sweep.
    #[test]
    fn idct_simd_1d_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let mut seed = 0x9E3779B9u32;
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for _ in 0..200 {
                let mut rows = [[0i64; 32]; 8];
                for row in rows.iter_mut() {
                    for v in row[..n0].iter_mut() {
                        *v = (xorshift32(&mut seed) & 0x7FF) as i64 - 1024;
                    }
                }
                let mut scalar = rows;
                for row in scalar.iter_mut() {
                    crate::transform::idct_permute(&mut row[..n0], n);
                    crate::transform::idct(&mut row[..n0], n);
                }
                let mut t = [unsafe { _mm256_setzero_si256() }; 32];
                for (k, tk) in t[..n0].iter_mut().enumerate() {
                    let lanes: [i32; 8] = std::array::from_fn(|r| rows[r][k] as i32);
                    *tk = unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) };
                }
                unsafe {
                    idct_permute_simd(&mut t, n);
                    idct_simd::<false>(&mut t, n);
                }
                for (k, &tk) in t[..n0].iter().enumerate() {
                    let mut lanes = [0i32; 8];
                    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, tk) };
                    for (r, &lane) in lanes.iter().enumerate() {
                        assert_eq!(lane as i64, scalar[r][k], "n={n}: row {r}, elem {k}");
                    }
                }
            }
        }
    }

    /// The fused `inverse_transform_dct_dct_reconstruct_avx2` (2D transform + per-pixel residual
    /// add + 8-bit clip) must exactly reproduce the scalar `inverse_transform_block` (DctDct)
    /// followed by the scalar `clip(pred + residual, 0, 255)` reconstruction, for every size
    /// n = 2..=5. Inputs are kept small (±64) so no transform intermediate exceeds i32 even in
    /// the worst case, isolating the transpose/driver/mirror/reconstruction structure; the
    /// i32-vs-i64 agreement on realistic (spec §8.7.1.1 16-bit-bounded) magnitudes is proven
    /// end-to-end by the official sweep.
    #[test]
    fn inverse_transform_dct_dct_reconstruct_simd_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let mut seed = 0x2468ACE0u32;
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            let count = n0 * n0;
            for _ in 0..100 {
                let mut dq = vec![0i64; count];
                for v in dq.iter_mut() {
                    *v = (xorshift32(&mut seed) & 0x7F) as i64 - 64;
                }
                let mut dq_s = dq.clone();
                crate::transform::inverse_transform_block(
                    &mut dq_s,
                    n,
                    crate::transform::TxType::DctDct,
                    false,
                );
                // Two placements: (0, 0) in a tight stride-n0 plane, and a nonzero origin
                // inside a wider plane -- the offset/stride seam of `reconstruct_add_clip`'s
                // raw `(start_y + i) * plane_width + start_x` indexing (exactly what the
                // tile-parallel column strips feed it). Whole-plane equality vs scalar also
                // proves no out-of-block writes.
                for &(start_x, start_y, plane_width) in &[(0usize, 0usize, n0), (4, 2, n0 + 8)] {
                    let pred: Vec<u16> = (0..plane_width * (start_y + n0))
                        .map(|_| (xorshift32(&mut seed) & 0xFF) as u16)
                        .collect();
                    let mut plane_scalar = pred.clone();
                    for i in 0..n0 {
                        for j in 0..n0 {
                            let idx = (start_y + i) * plane_width + start_x + j;
                            let old = plane_scalar[idx] as i64;
                            plane_scalar[idx] = (old + dq_s[i * n0 + j]).clamp(0, 255) as u16;
                        }
                    }

                    let mut plane_simd = pred.clone();
                    unsafe {
                        inverse_transform_dct_dct_reconstruct_avx2(
                            &mut plane_simd,
                            plane_width,
                            start_x,
                            start_y,
                            &dq,
                            n,
                        );
                    }
                    assert_eq!(
                        plane_scalar, plane_simd,
                        "n={n} at ({start_x},{start_y}) stride {plane_width}: fused SIMD != scalar"
                    );
                }
            }
        }
    }

    /// The HBD 1D inverse DCT (`idct_simd::<true>`, i.e. the `b_op_simd_hbd` butterfly) must
    /// exactly reproduce the scalar i64 `transform::idct_permute` + `transform::idct` on each
    /// lane, for every size n = 2..=5, at full 12-bit magnitudes: inputs span ±2^19 (the spec
    /// §8.7.1.1 `8 + BitDepth` bound), where `t * cos64` products reach ~2^33 -- beyond i32 --
    /// so this genuinely exercises the widened multiplies. (All STORED values stay inside i32:
    /// the 1D network's L2 gain is ~2^3 and L-inf <= sqrt(32) * L2, keeping intermediates under
    /// ~2^25.) As a self-check that these magnitudes discriminate, the 8-bit (`mullo`) network
    /// must diverge from the scalar somewhere across the run.
    #[test]
    fn idct_simd_hbd_1d_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let mut seed = 0xB5297A4Du32;
        let mut mullo_diverged = false;
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for _ in 0..200 {
                let mut rows = [[0i64; 32]; 8];
                for row in rows.iter_mut() {
                    for v in row[..n0].iter_mut() {
                        *v = (xorshift32(&mut seed) & 0xF_FFFF) as i64 - (1 << 19);
                    }
                }
                let mut scalar = rows;
                for row in scalar.iter_mut() {
                    crate::transform::idct_permute(&mut row[..n0], n);
                    crate::transform::idct(&mut row[..n0], n);
                }
                let mut t = [unsafe { _mm256_setzero_si256() }; 32];
                for (k, tk) in t[..n0].iter_mut().enumerate() {
                    let lanes: [i32; 8] = std::array::from_fn(|r| rows[r][k] as i32);
                    *tk = unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) };
                }
                let mut t_mullo = t;
                unsafe {
                    idct_permute_simd(&mut t, n);
                    idct_simd::<true>(&mut t, n);
                    idct_permute_simd(&mut t_mullo, n);
                    idct_simd::<false>(&mut t_mullo, n);
                }
                for (k, (&tk, &mk)) in t[..n0].iter().zip(t_mullo[..n0].iter()).enumerate() {
                    let mut lanes = [0i32; 8];
                    let mut lanes_mullo = [0i32; 8];
                    unsafe {
                        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, tk);
                        _mm256_storeu_si256(lanes_mullo.as_mut_ptr() as *mut __m256i, mk);
                    }
                    for (r, (&lane, &lane_mullo)) in
                        lanes.iter().zip(lanes_mullo.iter()).enumerate()
                    {
                        assert_eq!(lane as i64, scalar[r][k], "n={n}: row {r}, elem {k}");
                        mullo_diverged |= lane_mullo as i64 != scalar[r][k];
                    }
                }
            }
        }
        assert!(
            mullo_diverged,
            "test inputs never overflowed the i32-product network -- magnitudes too small to \
             exercise the widened HBD path"
        );
    }

    /// The fused HBD `inverse_transform_dct_dct_reconstruct_hbd_avx2` (2D transform + residual
    /// add + `(1 << bit_depth) - 1` clip) must exactly reproduce the scalar
    /// `inverse_transform_block` (DctDct) followed by the scalar `clamp(pred + residual, 0,
    /// max_val)` reconstruction, for every size n = 2..=5 at 10-bit AND 12-bit. Dequant inputs
    /// span ±2^(bit_depth+5) -- large enough that the column pass's butterfly products routinely
    /// exceed i32 at 12-bit (exercising the widened path end-to-end) while every stored
    /// intermediate stays inside the i32 lanes (the two-pass L2-gain bound keeps them under
    /// ~2^28); predictions span the full `0..(1 << bit_depth)` sample range. ±2^(bit_depth+5)
    /// suffices because the 1D network is already pinned at the full signed-`8 + BitDepth`-bit
    /// input domain by `idct_simd_hbd_1d_matches_scalar`, and everything the 2D driver adds on
    /// top of the two 1D passes (transposes, the final `round2` shift, the reconstruction) is
    /// structural, not magnitude-sensitive -- this test's job is that composition, not the
    /// arithmetic extremes.
    #[test]
    fn inverse_transform_dct_dct_reconstruct_hbd_simd_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let mut seed = 0x0DDB1A5Eu32;
        for &bit_depth in &[10u8, 12] {
            let max_val = (1i64 << bit_depth) - 1;
            let dq_span = 1u32 << (bit_depth + 6); // dequant values in ±2^(bit_depth+5)
            for n in 2..=5u32 {
                let n0 = 1usize << n;
                let count = n0 * n0;
                for _ in 0..100 {
                    let mut dq = vec![0i64; count];
                    for v in dq.iter_mut() {
                        *v = (xorshift32(&mut seed) % dq_span) as i64 - (dq_span / 2) as i64;
                    }
                    let mut dq_s = dq.clone();
                    crate::transform::inverse_transform_block(
                        &mut dq_s,
                        n,
                        crate::transform::TxType::DctDct,
                        false,
                    );
                    // Two placements (see the 8-bit DCT test): (0, 0)/stride n0, plus a
                    // nonzero origin in a wider plane covering `reconstruct_add_clip`'s
                    // offset/stride indexing -- for HBD nothing else covers that seam.
                    // Whole-plane equality also proves no out-of-block writes.
                    for &(start_x, start_y, plane_width) in &[(0usize, 0usize, n0), (4, 2, n0 + 8)]
                    {
                        let pred: Vec<u16> = (0..plane_width * (start_y + n0))
                            .map(|_| (xorshift32(&mut seed) & (max_val as u32)) as u16)
                            .collect();
                        let mut plane_scalar = pred.clone();
                        for i in 0..n0 {
                            for j in 0..n0 {
                                let idx = (start_y + i) * plane_width + start_x + j;
                                let old = plane_scalar[idx] as i64;
                                plane_scalar[idx] =
                                    (old + dq_s[i * n0 + j]).clamp(0, max_val) as u16;
                            }
                        }

                        let mut plane_simd = pred.clone();
                        unsafe {
                            inverse_transform_dct_dct_reconstruct_hbd_avx2(
                                &mut plane_simd,
                                plane_width,
                                start_x,
                                start_y,
                                &dq,
                                n,
                                bit_depth,
                            );
                        }
                        assert_eq!(
                            plane_scalar, plane_simd,
                            "bit_depth={bit_depth} n={n} at ({start_x},{start_y}) stride \
                             {plane_width}: fused HBD SIMD != scalar"
                        );
                    }
                }
            }
        }
    }

    /// The vector 1D inverse ADST (`iadst_simd`) must exactly reproduce the scalar
    /// `transform::iadst4/8/16` on each of its 8 lanes. n == 2 runs at the FULL spec §8.7.1.1
    /// input bound (`8 + BitDepth == 16` signed bits, -32768..=32767): every iadst4
    /// intermediate provably fits i32 there (peak chain 43801 * 2^15 < 2^31 -- see the
    /// inverse-ADST section comment). n == 3/4 use structure-validating ±512 inputs: an
    /// adversarial full-range VECTOR is not reachable from a conformant stream (each SB/SH
    /// stage re-bounds stored T values to 16 bits, which random full-range stage inputs
    /// violate) and would overflow i32 in ways conformant data cannot. The per-op spec-bound
    /// coverage is `adst_sb_sh_ops_match_scalar_at_spec_bounds`; the end-to-end
    /// conformant-magnitude proof is the official sweep.
    #[test]
    fn iadst_simd_1d_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let mut seed = 0xC0FF_EE11u32;
        for n in 2..=4u32 {
            let n0 = 1usize << n;
            for _ in 0..200 {
                let mut rows = [[0i64; 16]; 8];
                for row in rows.iter_mut() {
                    for v in row[..n0].iter_mut() {
                        *v = if n == 2 {
                            (xorshift32(&mut seed) & 0xFFFF) as i64 - 32768
                        } else {
                            (xorshift32(&mut seed) & 0x3FF) as i64 - 512
                        };
                    }
                }
                let mut scalar = rows;
                for row in scalar.iter_mut() {
                    match n {
                        2 => crate::transform::iadst4((&mut row[..4]).try_into().unwrap()),
                        3 => crate::transform::iadst8((&mut row[..8]).try_into().unwrap()),
                        _ => crate::transform::iadst16((&mut row[..16]).try_into().unwrap()),
                    }
                }
                let mut t = [unsafe { _mm256_setzero_si256() }; 32];
                for (k, tk) in t[..n0].iter_mut().enumerate() {
                    let lanes: [i32; 8] = std::array::from_fn(|r| rows[r][k] as i32);
                    *tk = unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) };
                }
                unsafe { iadst_simd(&mut t, n) };
                for (k, &tk) in t[..n0].iter().enumerate() {
                    let mut lanes = [0i32; 8];
                    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, tk) };
                    for (r, &lane) in lanes.iter().enumerate() {
                        assert_eq!(lane as i64, scalar[r][k], "n={n}: row {r}, elem {k}");
                    }
                }
            }
        }
    }

    /// `sb_op_simd` + `sh_op_simd` -- the ADST ops carrying the load-bearing i32 overflow
    /// claims -- must exactly match the spec's i64 arithmetic at the FULL spec §8.7.1.1 T
    /// bound (-32768..=32767, with the exact corners injected), for every rotation angle the
    /// iadst8/16 networks use, in both flip orientations. The SH inputs are the SB outputs of
    /// two INDEPENDENT worst-case pairs, so the `S[a] +/- S[b]` sums are exercised at their
    /// true maxima (|S| <= 2^15 * 23170 < 2^30, sums < 2^31).
    #[test]
    fn adst_sb_sh_ops_match_scalar_at_spec_bounds() {
        if !avx2_enabled() {
            return;
        }
        // Union of the `SB` angles in `transform.rs::iadst8_impl` / `iadst16_impl`.
        let angles = [30, 22, 14, 6, 24, 8, 31, 27, 23, 19, 15, 11, 7, 3, 28, 12];
        let mut seed = 0xAD57_0B57u32;
        for &angle in &angles {
            for &flip in &[false, true] {
                for trial in 0..100 {
                    // vals = [ta, tb, tc, td], each 8 lanes in the full signed-16-bit range;
                    // trial 0 plants the exact corners in the first two lanes.
                    let mut vals = [[0i64; 8]; 4];
                    for (vi, v) in vals.iter_mut().enumerate() {
                        for (lane, x) in v.iter_mut().enumerate() {
                            *x = if trial == 0 && lane < 2 {
                                if (lane + vi) % 2 == 0 {
                                    -32768
                                } else {
                                    32767
                                }
                            } else {
                                (xorshift32(&mut seed) & 0xFFFF) as i64 - 32768
                            };
                        }
                    }

                    // Scalar i64 reference: SB on (ta, tb) -> s0/s1, SB on (tc, td) -> s2/s3
                    // (spec §8.7.1.1's SB, with flip), then SH across the two rotations.
                    let cos = crate::transform::cos64(angle);
                    let sin = crate::transform::sin64(angle);
                    let round2_14_i64 = |x: i64| (x + (1i64 << 13)) >> 14;
                    let mut expected = [[0i64; 8]; 4];
                    for lane in 0..8 {
                        let (ta, tb) = (vals[0][lane], vals[1][lane]);
                        let (tc, td) = (vals[2][lane], vals[3][lane]);
                        let (mut s0, mut s1) = (ta * cos - tb * sin, ta * sin + tb * cos);
                        let (mut s2, mut s3) = (tc * cos - td * sin, tc * sin + td * cos);
                        if flip {
                            std::mem::swap(&mut s0, &mut s1);
                            std::mem::swap(&mut s2, &mut s3);
                        }
                        expected[0][lane] = round2_14_i64(s0 + s2);
                        expected[2][lane] = round2_14_i64(s0 - s2);
                        expected[1][lane] = round2_14_i64(s1 + s3);
                        expected[3][lane] = round2_14_i64(s1 - s3);
                    }

                    let load = |v: &[i64; 8]| {
                        let lanes: [i32; 8] = std::array::from_fn(|i| v[i] as i32);
                        unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) }
                    };
                    let mut t = [
                        load(&vals[0]),
                        load(&vals[1]),
                        load(&vals[2]),
                        load(&vals[3]),
                    ];
                    let mut s = [unsafe { _mm256_setzero_si256() }; 4];
                    unsafe {
                        sb_op_simd(&mut s, &t, 0, 1, angle, flip);
                        sb_op_simd(&mut s, &t, 2, 3, angle, flip);
                        sh_op_simd(&mut t, &s, 0, 2);
                        sh_op_simd(&mut t, &s, 1, 3);
                    }
                    for (k, &tk) in t.iter().enumerate() {
                        let mut lanes = [0i32; 8];
                        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, tk) };
                        for (lane, &got) in lanes.iter().enumerate() {
                            assert_eq!(
                                got as i64, expected[k][lane],
                                "angle={angle} flip={flip} trial={trial}: t[{k}] lane {lane}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `b_op_simd` / `b_op_simd_hbd` -- the `B` butterfly shared by the DCT and ADST networks,
    /// whose i32-`mullo` (8-bit) and widened-i64-product (10/12-bit) exactness claims are
    /// load-bearing -- must exactly match the spec's i64 arithmetic at the FULL spec §8.7.1.1
    /// stored-T bound for its depth (signed 16 bits for `b_op_simd`; the signed-20-bit
    /// 12-bit-depth bound for `b_op_simd_hbd`, where the products genuinely exceed i32), with
    /// the exact corners injected, for every rotation angle either network uses, in both flip
    /// orientations. Per-op companion to `adst_sb_sh_ops_match_scalar_at_spec_bounds` (SB/SH
    /// were previously the only ops tested at their bound).
    #[test]
    fn b_op_matches_scalar_at_spec_bounds() {
        if !avx2_enabled() {
            return;
        }
        // Union of every `B(a, b, angle, flip)` angle in the vector networks: `idct_simd` --
        // the n == 2 base and final-stage angle 16; `32 - brev(5, a)` over the recursion
        // levels (n=3: 28,12; n=4: 30,14,22,6; n=5: 31,15,23,7,27,11,19,3); the n == 5 stage
        // `28 - 16i + 56j` (84,68 beyond the above); the n >= 4 stage `24 + 48j` (24,72) --
        // plus `iadst8_impl` (16) and `iadst16_impl` (`48 + 64*(i^j)`: 48,112).
        let angles = [
            3, 6, 7, 11, 12, 14, 15, 16, 19, 22, 23, 24, 27, 28, 30, 31, 48, 68, 72, 84, 112,
        ];
        let mut seed = 0xB0B0_5EEDu32;
        for &(hbd, bound) in &[(false, 1i64 << 15), (true, 1i64 << 19)] {
            for &angle in &angles {
                for &flip in &[false, true] {
                    for trial in 0..100 {
                        // vals = [ta, tb], each 8 lanes across the full signed range
                        // [-bound, bound); trial 0 plants the exact corners in the first
                        // two lanes.
                        let mut vals = [[0i64; 8]; 2];
                        for (vi, v) in vals.iter_mut().enumerate() {
                            for (lane, x) in v.iter_mut().enumerate() {
                                *x = if trial == 0 && lane < 2 {
                                    if (lane + vi) % 2 == 0 {
                                        -bound
                                    } else {
                                        bound - 1
                                    }
                                } else {
                                    (xorshift32(&mut seed) as i64 & (2 * bound - 1)) - bound
                                };
                            }
                        }

                        // Scalar i64 reference (`transform.rs::b_op`, spec §8.7.1.1's B):
                        // x = ta*cos - tb*sin, y = ta*sin + tb*cos, round2(., 14), flip-swap.
                        let cos = crate::transform::cos64(angle);
                        let sin = crate::transform::sin64(angle);
                        let round2_14_i64 = |x: i64| (x + (1i64 << 13)) >> 14;
                        let mut expected = [[0i64; 8]; 2];
                        for lane in 0..8 {
                            let (ta, tb) = (vals[0][lane], vals[1][lane]);
                            let (mut na, mut nb) = (
                                round2_14_i64(ta * cos - tb * sin),
                                round2_14_i64(ta * sin + tb * cos),
                            );
                            if flip {
                                std::mem::swap(&mut na, &mut nb);
                            }
                            expected[0][lane] = na;
                            expected[1][lane] = nb;
                        }

                        let load = |v: &[i64; 8]| {
                            let lanes: [i32; 8] = std::array::from_fn(|i| v[i] as i32);
                            unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) }
                        };
                        let mut t = [load(&vals[0]), load(&vals[1])];
                        unsafe {
                            if hbd {
                                b_op_simd_hbd(&mut t, 0, 1, angle, flip);
                            } else {
                                b_op_simd(&mut t, 0, 1, angle, flip);
                            }
                        }
                        for (k, &tk) in t.iter().enumerate() {
                            let mut lanes = [0i32; 8];
                            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, tk) };
                            for (lane, &got) in lanes.iter().enumerate() {
                                assert_eq!(
                                    got as i64, expected[k][lane],
                                    "hbd={hbd} angle={angle} flip={flip} trial={trial}: \
                                     t[{k}] lane {lane}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The fused `inverse_transform_adst_reconstruct_avx2` (2D transform + residual add + 8-bit
    /// clip) must exactly reproduce the scalar `inverse_transform_block` + `clip(pred +
    /// residual, 0, 255)`, for every ADST-containing tx type and every ADST size (4/8/16).
    /// n == 2 runs dequant magnitudes of ±8192 -- NOT the conformant input ceiling (which is
    /// ±2^15, covered per-pass by the full-range 1D test `iadst_simd_1d_matches_scalar`), but
    /// the largest range for which the 4x4 2D composition is arithmetically i32-safe without
    /// invoking any conformance bound between the passes (worst chain 43801 * (2.73 * 8192)
    /// < 2^31). n == 3/4 use ±32 (like the DCT 2D test: small enough that no intermediate can
    /// overflow i32 even with adversarial -- not-conformant -- magnitude compounding across
    /// the two passes; the conformant-magnitude proof is the official sweep).
    #[test]
    fn inverse_transform_adst_reconstruct_simd_matches_scalar() {
        if !avx2_enabled() {
            return;
        }
        let mut seed = 0x5EED_AD57u32;
        for &tx_type in &[
            crate::transform::TxType::AdstDct,
            crate::transform::TxType::DctAdst,
            crate::transform::TxType::AdstAdst,
        ] {
            for n in 2..=4u32 {
                let n0 = 1usize << n;
                let count = n0 * n0;
                let span: u32 = if n == 2 { 16384 } else { 64 }; // dequant in ±8192 / ±32
                for _ in 0..100 {
                    let mut dq = vec![0i64; count];
                    for v in dq.iter_mut() {
                        *v = (xorshift32(&mut seed) % span) as i64 - (span / 2) as i64;
                    }
                    let mut dq_s = dq.clone();
                    crate::transform::inverse_transform_block(&mut dq_s, n, tx_type, false);
                    // Two placements (see the 8-bit DCT test): (0, 0)/stride n0, plus a
                    // nonzero origin in a wider plane covering `reconstruct_add_clip`'s
                    // offset/stride indexing. Whole-plane equality also proves no
                    // out-of-block writes.
                    for &(start_x, start_y, plane_width) in &[(0usize, 0usize, n0), (4, 2, n0 + 8)]
                    {
                        let pred: Vec<u16> = (0..plane_width * (start_y + n0))
                            .map(|_| (xorshift32(&mut seed) & 0xFF) as u16)
                            .collect();
                        let mut plane_scalar = pred.clone();
                        for i in 0..n0 {
                            for j in 0..n0 {
                                let idx = (start_y + i) * plane_width + start_x + j;
                                let old = plane_scalar[idx] as i64;
                                plane_scalar[idx] = (old + dq_s[i * n0 + j]).clamp(0, 255) as u16;
                            }
                        }

                        let mut plane_simd = pred.clone();
                        unsafe {
                            inverse_transform_adst_reconstruct_avx2(
                                &mut plane_simd,
                                plane_width,
                                start_x,
                                start_y,
                                &dq,
                                n,
                                tx_type,
                            );
                        }
                        assert_eq!(
                            plane_scalar, plane_simd,
                            "tx_type={tx_type:?} n={n} at ({start_x},{start_y}) stride \
                             {plane_width}: fused ADST SIMD != scalar"
                        );
                    }
                }
            }
        }
    }
}
