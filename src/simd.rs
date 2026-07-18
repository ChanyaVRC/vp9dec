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
