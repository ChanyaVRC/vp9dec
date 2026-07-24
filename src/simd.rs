//! AVX2 SIMD kernels, x86_64-only (`core::arch::x86_64` intrinsics, zero dependencies),
//! runtime-detected and force-disabled via [`avx2_enabled`]. The scalar decode paths own
//! every dispatch point and fallback; this module owns only the vector kernels, split by
//! pipeline stage into submodules (all entry points re-exported here, so call sites use
//! `crate::simd::<kernel>`):
//!
//! - `inter`: the inter-prediction two-pass 8-tap subpel convolution (spec §8.5.2.4),
//!   unscaled (widths 8+ and 4) and scaled-reference (SVC / resize); `predict.rs`
//!   dispatches.
//! - `loop_filter`: the deblocking edge filters (spec §8.8.5, narrow / wide8 / wide16) for
//!   both edge orientations; `loop_filter.rs` dispatches.
//! - `transform`: the inverse DCT/ADST transforms fused with reconstruction (spec §8.7);
//!   `tile/residual.rs` dispatches (DCT_DCT and the ADST-containing types at all bit
//!   depths; only lossless WHT stays scalar -- see the landmines in
//!   docs/implementation-notes.md).
//!
//! A NEON mirror for aarch64 would be sibling `#[cfg(target_arch = "aarch64")]` modules
//! behind the same dispatch points -- not implemented.

mod inter;
mod loop_filter;
mod transform;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;

pub use inter::{
    block_inter_predict_avx2, block_inter_predict_avx2_w4, block_inter_predict_scaled_avx2,
};
pub use loop_filter::{loop_filter_horiz8_avx2, loop_filter_vert8_avx2};
pub use transform::{
    inverse_transform_adst_reconstruct_avx2, inverse_transform_adst_reconstruct_hbd_avx2,
    inverse_transform_dct_dct_reconstruct_avx2, inverse_transform_dct_dct_reconstruct_hbd_avx2,
};

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
