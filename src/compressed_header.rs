//! Parsing of the compressed header (`compressed_header`) (spec §6.3).
//!
//! `compressed_header()` is `header_size_in_bytes` bytes of bool-coded data
//! holding the transform mode (`tx_mode`) and the update contents of the
//! various probability tables.
//!
//! ```text
//! compressed_header( ) {
//!     read_tx_mode( )
//!     if ( tx_mode == TX_MODE_SELECT ) {
//!        tx_mode_probs( )
//!     }
//!     read_coef_probs( )
//!     read_skip_prob( )
//!     if ( FrameIsIntra == 0 ) {
//!        read_inter_mode_probs( )
//!        if ( interpolation_filter == SWITCHABLE ) read_interp_filter_probs( )
//!        read_is_inter_probs( )
//!        frame_reference_mode( )
//!        frame_reference_mode_probs( )
//!        read_y_mode_probs( )
//!        read_partition_probs( )
//!        mv_probs( )
//!     }
//! }
//! ```
//!
//! The inter-related reads called only when `FrameIsIntra == 0`
//! (`read_inter_mode_probs` onward, spec §6.3.9-6.3.18) were implemented in M3.

use crate::bool_coder::{BoolCoderError, BoolDecoder};
use crate::prob_tables::{
    CoefProbs, ALTREF_FRAME, COMPOUND_REFERENCE, DEFAULT_COEF_PROBS, DEFAULT_COMP_MODE_PROB,
    DEFAULT_COMP_REF_PROB, DEFAULT_INTERP_FILTER_PROBS, DEFAULT_INTER_MODE_PROBS,
    DEFAULT_IS_INTER_PROB, DEFAULT_MV_BITS_PROB, DEFAULT_MV_CLASS0_BIT_PROB,
    DEFAULT_MV_CLASS0_FR_PROBS, DEFAULT_MV_CLASS0_HP_PROB, DEFAULT_MV_CLASS_PROBS,
    DEFAULT_MV_FR_PROBS, DEFAULT_MV_HP_PROB, DEFAULT_MV_JOINT_PROBS, DEFAULT_MV_SIGN_PROB,
    DEFAULT_PARTITION_PROBS, DEFAULT_SINGLE_REF_PROB, DEFAULT_SKIP_PROB, DEFAULT_TX_PROBS,
    DEFAULT_UV_MODE_PROBS, DEFAULT_Y_MODE_PROBS, GOLDEN_FRAME, INV_MAP_TABLE, LAST_FRAME,
    REFERENCE_MODE_SELECT, SINGLE_REFERENCE, SWITCHABLE, TX_16X16, TX_32X32, TX_4X4, TX_8X8,
    TX_MODE_SELECT, TX_MODE_TO_BIGGEST_TX_SIZE,
};

/// Errors that can occur while parsing `compressed_header`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressedHeaderError {
    /// Bool decoder initialization failed (e.g. `header_size_in_bytes` is 0).
    BoolCoder(BoolCoderError),
}

/// The full set of probability tables updated by `compressed_header()`.
///
/// Corresponds to "all probability tables" operated on by the spec's
/// `load_probs`/`save_probs` (spec §7.1.2), and is saved/restored as-is as the
/// frame context ([`FrameContext`], 4 slots).
///
/// `uv_mode_probs` has no forward update syntax in `compressed_header()`
/// (`read_y_mode_probs()` only updates `y_mode_probs`), but it is a target of
/// backward adaptation in spec §8.4.4 `adapt_noncoef_probs()`
/// (`adapt_probs( intra_mode_tree, uv_mode_probs[ i ], counts_uv_mode[ i ] )`),
/// so it is kept here as one of the tables operated on by `load_probs`/`save_probs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedHeaderProbs {
    /// `uv_mode_probs[y_mode][node]`. Has no forward update syntax but is a
    /// target of backward adaptation (see doc comment above).
    pub uv_mode_probs: [[u8; 9]; 10],
    /// `tx_probs[maxTxSize][ctx][node]`. Same layout as [`crate::prob_tables::DEFAULT_TX_PROBS`].
    pub tx_probs: [[[u8; 3]; 2]; 4],
    /// `coef_probs[txSz][plane>0][is_inter][band][ctx][node]`.
    pub coef_probs: CoefProbs,
    /// `skip_prob[ctx]` (spec §6.3.8).
    pub skip_prob: [u8; 3],
    /// `inter_mode_probs[ctx][node]` (spec §6.3.9). Updated only when `FrameIsIntra == 0`.
    pub inter_mode_probs: [[u8; 3]; 7],
    /// `interp_filter_probs[ctx][node]` (spec §6.3.10).
    pub interp_filter_probs: [[u8; 2]; 4],
    /// `is_inter_prob[ctx]` (spec §6.3.11).
    pub is_inter_prob: [u8; 4],
    /// `comp_mode_prob[ctx]` (spec §6.3.13).
    pub comp_mode_prob: [u8; 5],
    /// `single_ref_prob[ctx][0..2]` (spec §6.3.13).
    pub single_ref_prob: [[u8; 2]; 5],
    /// `comp_ref_prob[ctx]` (spec §6.3.13).
    pub comp_ref_prob: [u8; 5],
    /// `y_mode_probs[ctx][node]` (spec §6.3.14). Non-key-frame only (key
    /// frames always use the fixed table [`crate::prob_tables::KF_Y_MODE_PROBS`]).
    pub y_mode_probs: [[u8; 9]; 4],
    /// `partition_probs[ctx][node]` (spec §6.3.15). Non-key-frame only (key
    /// frames always use the fixed table [`crate::prob_tables::KF_PARTITION_PROBS`]).
    pub partition_probs: [[u8; 3]; 16],
    /// `mv_joint_probs[node]` (spec §6.3.16).
    pub mv_joint_probs: [u8; 3],
    /// `mv_sign_prob[comp]`.
    pub mv_sign_prob: [u8; 2],
    /// `mv_class_probs[comp][node]`.
    pub mv_class_probs: [[u8; 10]; 2],
    /// `mv_class0_bit_prob[comp]`.
    pub mv_class0_bit_prob: [u8; 2],
    /// `mv_bits_prob[comp][i]`.
    pub mv_bits_prob: [[u8; 10]; 2],
    /// `mv_class0_fr_probs[comp][class0bit][node]`.
    pub mv_class0_fr_probs: [[[u8; 3]; 2]; 2],
    /// `mv_fr_probs[comp][node]`.
    pub mv_fr_probs: [[u8; 3]; 2],
    /// `mv_class0_hp_prob[comp]`.
    pub mv_class0_hp_prob: [u8; 2],
    /// `mv_hp_prob[comp]`.
    pub mv_hp_prob: [u8; 2],
}

impl Default for CompressedHeaderProbs {
    fn default() -> Self {
        Self {
            uv_mode_probs: DEFAULT_UV_MODE_PROBS,
            tx_probs: DEFAULT_TX_PROBS,
            coef_probs: DEFAULT_COEF_PROBS,
            skip_prob: DEFAULT_SKIP_PROB,
            inter_mode_probs: DEFAULT_INTER_MODE_PROBS,
            interp_filter_probs: DEFAULT_INTERP_FILTER_PROBS,
            is_inter_prob: DEFAULT_IS_INTER_PROB,
            comp_mode_prob: DEFAULT_COMP_MODE_PROB,
            single_ref_prob: DEFAULT_SINGLE_REF_PROB,
            comp_ref_prob: DEFAULT_COMP_REF_PROB,
            y_mode_probs: DEFAULT_Y_MODE_PROBS,
            partition_probs: DEFAULT_PARTITION_PROBS,
            mv_joint_probs: DEFAULT_MV_JOINT_PROBS,
            mv_sign_prob: DEFAULT_MV_SIGN_PROB,
            mv_class_probs: DEFAULT_MV_CLASS_PROBS,
            mv_class0_bit_prob: DEFAULT_MV_CLASS0_BIT_PROB,
            mv_bits_prob: DEFAULT_MV_BITS_PROB,
            mv_class0_fr_probs: DEFAULT_MV_CLASS0_FR_PROBS,
            mv_fr_probs: DEFAULT_MV_FR_PROBS,
            mv_class0_hp_prob: DEFAULT_MV_CLASS0_HP_PROB,
            mv_hp_prob: DEFAULT_MV_HP_PROB,
        }
    }
}

/// Frame context (the unit operated on by `load_probs`/`save_probs` in spec §7.1.2).
/// `CompressedHeaderProbs` itself corresponds to "all probability tables that are saved and restored".
pub type FrameContext = CompressedHeaderProbs;

/// The 4-slot frame context storage area, addressed by `frame_context_idx` (0..=3).
///
/// Spec §7.2's `setup_past_independence()` resets all probability tables to
/// their default values for key frames, intra-only frames, and error-resilient
/// frames, and then (under conditions such as `frame_type == KEY_FRAME`) calls
/// `save_probs(i)` for all 4 slots. For non-key frames, `load_probs` is
/// performed from the slot pointed to by `frame_context_idx` (the
/// `starting_probs` argument of `parse_compressed_header_ex` in this decoder),
/// and if `refresh_frame_context == 1`, the result is written back to the same
/// slot via `save_probs`.
///
/// Backward probability adaptation from spec §8.4
/// (`adapt_coef_probs`/`adapt_noncoef_probs`, based on observed frequencies)
/// is implemented in `counts.rs` and driven by `Decoder` (see
/// `refresh_probs` in `lib.rs`), which adapts the probabilities after tile
/// decode and writes them back via `save_probs` when
/// `refresh_frame_context == 1`.
#[derive(Debug, Clone)]
pub struct FrameContextStore {
    contexts: [FrameContext; 4],
}

impl FrameContextStore {
    /// Initializes all 4 slots with default values (equivalent to calling
    /// `save_probs(i)` for i in 0..4 right after `setup_past_independence()`).
    pub fn new() -> Self {
        Self {
            contexts: std::array::from_fn(|_| FrameContext::default()),
        }
    }

    pub fn load(&self, idx: u8) -> FrameContext {
        self.contexts[idx as usize].clone()
    }

    pub fn save(&mut self, idx: u8, ctx: FrameContext) {
        self.contexts[idx as usize] = ctx;
    }

    /// The all-slots reset performed by `setup_past_independence()` (occurs
    /// for key frames, error-resilient frames, etc.).
    pub fn reset_all(&mut self) {
        self.contexts = std::array::from_fn(|_| FrameContext::default());
    }
}

impl Default for FrameContextStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of parsing `compressed_header()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedHeader {
    /// `tx_mode` (spec §7.3.1).
    pub tx_mode: u8,
    /// The full set of updated probability tables.
    pub probs: CompressedHeaderProbs,
    /// `reference_mode` (spec §7.3.6). Always `SINGLE_REFERENCE` when `FrameIsIntra == 1`.
    pub reference_mode: u8,
    /// `CompFixedRef` (spec §6.3.18). Unused (0) when `reference_mode == SINGLE_REFERENCE`.
    pub comp_fixed_ref: u8,
    /// `CompVarRef[ 0..2 ]` (spec §6.3.18). Unused (`[0, 0]`) when
    /// `reference_mode == SINGLE_REFERENCE`.
    pub comp_var_ref: [u8; 2],
}

/// A `B(252)` bool read plus update decision, equivalent to `read_prob()` (spec §6.3.3 `diff_update_prob`).
///
/// ```text
/// diff_update_prob( prob ) {
///     update_prob                  B(252)
///     if ( update_prob == 1 ) {
///        deltaProb = decode_term_subexp( )
///        prob = inv_remap_prob( deltaProb, prob )
///     }
///     return prob
/// }
/// ```
fn diff_update_prob(r: &mut BoolDecoder, prob: u8) -> u8 {
    let update_prob = r.read_bool(252);
    if update_prob {
        let delta_prob = decode_term_subexp(r);
        inv_remap_prob(delta_prob, prob)
    } else {
        prob
    }
}

/// `decode_term_subexp()` (spec §6.3.4). All fields are read via `L(n)` (`read_literal`).
fn decode_term_subexp(r: &mut BoolDecoder) -> u32 {
    if r.read_literal(1) == 0 {
        return r.read_literal(4);
    }
    if r.read_literal(1) == 0 {
        return r.read_literal(4) + 16;
    }
    if r.read_literal(1) == 0 {
        return r.read_literal(5) + 32;
    }
    let v = r.read_literal(7);
    if v < 65 {
        return v + 64;
    }
    let bit = r.read_literal(1);
    (v << 1) - 1 + bit
}

/// `inv_remap_prob( deltaProb, prob )` (spec §6.3.5).
fn inv_remap_prob(delta_prob: u32, prob: u8) -> u8 {
    let v = INV_MAP_TABLE[delta_prob as usize] as u32;
    // m-- (prob minus 1) is used from here on.
    let m = prob as i32 - 1;
    let result = if (m << 1) <= 255 {
        1 + inv_recenter_nonneg(v, m as u32) as i32
    } else {
        255 - inv_recenter_nonneg(v, (255 - 1 - m) as u32) as i32
    };
    result as u8
}

/// `inv_recenter_nonneg( v, m )` (spec §6.3.6).
fn inv_recenter_nonneg(v: u32, m: u32) -> u32 {
    if v > 2 * m {
        return v;
    }
    if v & 1 == 1 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// `read_tx_mode()` (spec §6.3.1).
fn read_tx_mode(r: &mut BoolDecoder, lossless: bool) -> u8 {
    if lossless {
        TX_4X4 // Same value as ONLY_4X4 (0)
    } else {
        let mut tx_mode = r.read_literal(2) as u8;
        if tx_mode == TX_32X32 {
            // ALLOW_32X32 (=3) and TX_32X32 (=3) share the same value, so the same constant is reused.
            let tx_mode_select = r.read_literal(1) as u8;
            tx_mode += tx_mode_select;
        }
        tx_mode
    }
}

/// `tx_mode_probs()` (spec §6.3.2). Called only when `tx_mode == TX_MODE_SELECT`.
fn read_tx_mode_probs(r: &mut BoolDecoder, tx_probs: &mut [[[u8; 3]; 2]; 4]) {
    // tx_probs_8x8[ TX_SIZE_CONTEXTS ][ TX_SIZES - 3 = 1 ]
    for ctx in tx_probs[TX_8X8 as usize].iter_mut() {
        for node in ctx.iter_mut().take(1) {
            *node = diff_update_prob(r, *node);
        }
    }
    // tx_probs_16x16[ TX_SIZE_CONTEXTS ][ TX_SIZES - 2 = 2 ]
    for ctx in tx_probs[TX_16X16 as usize].iter_mut() {
        for node in ctx.iter_mut().take(2) {
            *node = diff_update_prob(r, *node);
        }
    }
    // tx_probs_32x32[ TX_SIZE_CONTEXTS ][ TX_SIZES - 1 = 3 ]
    for ctx in tx_probs[TX_32X32 as usize].iter_mut() {
        for node in ctx.iter_mut().take(3) {
            *node = diff_update_prob(r, *node);
        }
    }
}

/// `read_coef_probs()` (spec §6.3.7).
fn read_coef_probs(r: &mut BoolDecoder, tx_mode: u8, coef_probs: &mut CoefProbs) {
    let max_tx_size = TX_MODE_TO_BIGGEST_TX_SIZE[tx_mode as usize];
    for tx_sz in 0..=max_tx_size {
        let update_probs = r.read_literal(1) == 1;
        if !update_probs {
            continue;
        }
        for plane_probs in coef_probs[tx_sz as usize].iter_mut() {
            for ref_probs in plane_probs.iter_mut() {
                for (k, band_probs) in ref_probs.iter_mut().enumerate() {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for ctx_probs in band_probs.iter_mut().take(max_l) {
                        for prob in ctx_probs.iter_mut() {
                            *prob = diff_update_prob(r, *prob);
                        }
                    }
                }
            }
        }
    }
}

/// `read_skip_prob()` (spec §6.3.8).
fn read_skip_prob(r: &mut BoolDecoder, skip_prob: &mut [u8; 3]) {
    for prob in skip_prob.iter_mut() {
        *prob = diff_update_prob(r, *prob);
    }
}

/// `read_inter_mode_probs()` (spec §6.3.9).
fn read_inter_mode_probs(r: &mut BoolDecoder, probs: &mut [[u8; 3]; 7]) {
    for ctx in probs.iter_mut() {
        for node in ctx.iter_mut() {
            *node = diff_update_prob(r, *node);
        }
    }
}

/// `read_interp_filter_probs()` (spec §6.3.10).
fn read_interp_filter_probs(r: &mut BoolDecoder, probs: &mut [[u8; 2]; 4]) {
    for ctx in probs.iter_mut() {
        for node in ctx.iter_mut() {
            *node = diff_update_prob(r, *node);
        }
    }
}

/// `read_is_inter_probs()` (spec §6.3.11).
fn read_is_inter_probs(r: &mut BoolDecoder, probs: &mut [u8; 4]) {
    for prob in probs.iter_mut() {
        *prob = diff_update_prob(r, *prob);
    }
}

/// `setup_compound_reference_mode()` (spec §6.3.18). Returns `(CompFixedRef, CompVarRef)`.
fn setup_compound_reference_mode(ref_frame_sign_bias: &[bool; 4]) -> (u8, [u8; 2]) {
    if ref_frame_sign_bias[LAST_FRAME as usize] == ref_frame_sign_bias[GOLDEN_FRAME as usize] {
        (ALTREF_FRAME, [LAST_FRAME, GOLDEN_FRAME])
    } else if ref_frame_sign_bias[LAST_FRAME as usize] == ref_frame_sign_bias[ALTREF_FRAME as usize]
    {
        (GOLDEN_FRAME, [LAST_FRAME, ALTREF_FRAME])
    } else {
        (LAST_FRAME, [GOLDEN_FRAME, ALTREF_FRAME])
    }
}

/// `frame_reference_mode()` (spec §6.3.12). Returns
/// `(reference_mode, CompFixedRef, CompVarRef)`.
fn frame_reference_mode(r: &mut BoolDecoder, ref_frame_sign_bias: &[bool; 4]) -> (u8, u8, [u8; 2]) {
    let compound_reference_allowed = ref_frame_sign_bias[GOLDEN_FRAME as usize]
        != ref_frame_sign_bias[LAST_FRAME as usize]
        || ref_frame_sign_bias[ALTREF_FRAME as usize] != ref_frame_sign_bias[LAST_FRAME as usize];

    let reference_mode = if compound_reference_allowed {
        let non_single_reference = r.read_literal(1) == 1;
        if !non_single_reference {
            SINGLE_REFERENCE
        } else {
            let reference_select = r.read_literal(1) == 1;
            if !reference_select {
                COMPOUND_REFERENCE
            } else {
                REFERENCE_MODE_SELECT
            }
        }
    } else {
        SINGLE_REFERENCE
    };

    let (comp_fixed_ref, comp_var_ref) = if reference_mode != SINGLE_REFERENCE {
        setup_compound_reference_mode(ref_frame_sign_bias)
    } else {
        (0, [0, 0])
    };

    (reference_mode, comp_fixed_ref, comp_var_ref)
}

/// `frame_reference_mode_probs()` (spec §6.3.13).
fn frame_reference_mode_probs(
    r: &mut BoolDecoder,
    reference_mode: u8,
    probs: &mut CompressedHeaderProbs,
) {
    if reference_mode == REFERENCE_MODE_SELECT {
        for prob in probs.comp_mode_prob.iter_mut() {
            *prob = diff_update_prob(r, *prob);
        }
    }
    if reference_mode != COMPOUND_REFERENCE {
        for ctx in probs.single_ref_prob.iter_mut() {
            ctx[0] = diff_update_prob(r, ctx[0]);
            ctx[1] = diff_update_prob(r, ctx[1]);
        }
    }
    if reference_mode != SINGLE_REFERENCE {
        for prob in probs.comp_ref_prob.iter_mut() {
            *prob = diff_update_prob(r, *prob);
        }
    }
}

/// `read_y_mode_probs()` (spec §6.3.14). Updates the non-key-frame-only `y_mode_probs`.
fn read_y_mode_probs(r: &mut BoolDecoder, probs: &mut [[u8; 9]; 4]) {
    for ctx in probs.iter_mut() {
        for node in ctx.iter_mut() {
            *node = diff_update_prob(r, *node);
        }
    }
}

/// `read_partition_probs()` (spec §6.3.15). Updates the non-key-frame-only `partition_probs`.
fn read_partition_probs(r: &mut BoolDecoder, probs: &mut [[u8; 3]; 16]) {
    for ctx in probs.iter_mut() {
        for node in ctx.iter_mut() {
            *node = diff_update_prob(r, *node);
        }
    }
}

/// `update_mv_prob( prob )` (spec §6.3.17). Note that, unlike `diff_update_prob`,
/// after reading whether to update via `B(252)`, this reads `L(7)` directly
/// rather than using `decode_term_subexp`/`inv_remap_prob`.
fn update_mv_prob(r: &mut BoolDecoder, prob: u8) -> u8 {
    if r.read_bool(252) {
        let mv_prob = r.read_literal(7) as u8;
        (mv_prob << 1) | 1
    } else {
        prob
    }
}

/// `mv_probs()` (spec §6.3.16).
fn mv_probs(r: &mut BoolDecoder, allow_high_precision_mv: bool, probs: &mut CompressedHeaderProbs) {
    for prob in probs.mv_joint_probs.iter_mut() {
        *prob = update_mv_prob(r, *prob);
    }
    for i in 0..2 {
        probs.mv_sign_prob[i] = update_mv_prob(r, probs.mv_sign_prob[i]);
        for j in 0..probs.mv_class_probs[i].len() {
            probs.mv_class_probs[i][j] = update_mv_prob(r, probs.mv_class_probs[i][j]);
        }
        probs.mv_class0_bit_prob[i] = update_mv_prob(r, probs.mv_class0_bit_prob[i]);
        for j in 0..probs.mv_bits_prob[i].len() {
            probs.mv_bits_prob[i][j] = update_mv_prob(r, probs.mv_bits_prob[i][j]);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..probs.mv_class0_fr_probs[i][j].len() {
                probs.mv_class0_fr_probs[i][j][k] =
                    update_mv_prob(r, probs.mv_class0_fr_probs[i][j][k]);
            }
        }
        for k in 0..probs.mv_fr_probs[i].len() {
            probs.mv_fr_probs[i][k] = update_mv_prob(r, probs.mv_fr_probs[i][k]);
        }
    }
    if allow_high_precision_mv {
        for i in 0..2 {
            probs.mv_class0_hp_prob[i] = update_mv_prob(r, probs.mv_class0_hp_prob[i]);
            probs.mv_hp_prob[i] = update_mv_prob(r, probs.mv_hp_prob[i]);
        }
    }
}

/// Parses `compressed_header()` (spec §6.3) (a simplified key-frame-only wrapper).
///
/// `data` is the `header_size_in_bytes`-byte slice. `lossless` is the
/// `Lossless` flag obtained from the uncompressed header's
/// `quantization_params()`. Calls [`parse_compressed_header_ex`] with
/// `FrameIsIntra == 1` (starting probabilities are always the default values).
pub fn parse_compressed_header(
    data: &[u8],
    lossless: bool,
) -> Result<CompressedHeader, CompressedHeaderError> {
    parse_compressed_header_ex(
        data,
        lossless,
        true,
        SWITCHABLE,
        [false; 4],
        false,
        CompressedHeaderProbs::default(),
    )
}

/// Parses `compressed_header()` (spec §6.3) (the full version, supporting inter frames).
///
/// - `frame_is_intra`: `FrameIsIntra`. When true, none of the inter-related
///   syntax (spec §6.3.9-6.3.18) is read at all.
/// - `interpolation_filter`: the uncompressed header's `interpolation_filter`
///   (ignored when `FrameIsIntra == 1`).
/// - `ref_frame_sign_bias`: the uncompressed header's `ref_frame_sign_bias` (indexed by `ref_frame` value).
/// - `allow_high_precision_mv`: the uncompressed header's field of the same name.
/// - `starting_probs`: the starting probability table equivalent to
///   `load_probs( frame_context_idx )` (obtained via [`FrameContextStore::load`]).
#[allow(clippy::too_many_arguments)]
pub fn parse_compressed_header_ex(
    data: &[u8],
    lossless: bool,
    frame_is_intra: bool,
    interpolation_filter: u8,
    ref_frame_sign_bias: [bool; 4],
    allow_high_precision_mv: bool,
    starting_probs: CompressedHeaderProbs,
) -> Result<CompressedHeader, CompressedHeaderError> {
    let mut r = BoolDecoder::new(data).map_err(CompressedHeaderError::BoolCoder)?;
    let mut probs = starting_probs;

    let tx_mode = read_tx_mode(&mut r, lossless);
    if tx_mode == TX_MODE_SELECT {
        read_tx_mode_probs(&mut r, &mut probs.tx_probs);
    }
    read_coef_probs(&mut r, tx_mode, &mut probs.coef_probs);
    read_skip_prob(&mut r, &mut probs.skip_prob);

    let mut reference_mode = SINGLE_REFERENCE;
    let mut comp_fixed_ref = 0u8;
    let mut comp_var_ref = [0u8; 2];

    if !frame_is_intra {
        read_inter_mode_probs(&mut r, &mut probs.inter_mode_probs);
        if interpolation_filter == SWITCHABLE {
            read_interp_filter_probs(&mut r, &mut probs.interp_filter_probs);
        }
        read_is_inter_probs(&mut r, &mut probs.is_inter_prob);
        let (rm, cfr, cvr) = frame_reference_mode(&mut r, &ref_frame_sign_bias);
        reference_mode = rm;
        comp_fixed_ref = cfr;
        comp_var_ref = cvr;
        frame_reference_mode_probs(&mut r, reference_mode, &mut probs);
        read_y_mode_probs(&mut r, &mut probs.y_mode_probs);
        read_partition_probs(&mut r, &mut probs.partition_probs);
        mv_probs(&mut r, allow_high_precision_mv, &mut probs);
    }

    r.exit_bool();

    Ok(CompressedHeader {
        tx_mode,
        probs,
        reference_mode,
        comp_fixed_ref,
        comp_var_ref,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prob_tables::ONLY_4X4;
    use crate::test_support::BoolEncoder;

    #[test]
    fn lossless_frame_forces_only_4x4_and_reads_no_extra_bit() {
        // When lossless == true, tx_mode is always ONLY_4X4 and nothing is
        // read from the bitstream (read_tx_mode's if branch is skipped
        // entirely), so only the subsequent read_coef_probs (txSz is TX_4X4
        // only) and read_skip_prob are encoded.
        let mut enc = BoolEncoder::new();
        enc.write_literal(0, 1); // read_coef_probs: txSz=TX_4X4, update_probs=0
        enc.write_bool(false, 252); // read_skip_prob[0]: update_prob=0
        enc.write_bool(false, 252); // read_skip_prob[1]
        enc.write_bool(false, 252); // read_skip_prob[2]
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, true).expect("should parse");
        assert_eq!(header.tx_mode, ONLY_4X4);
        assert_eq!(header.probs, CompressedHeaderProbs::default());
    }

    #[test]
    fn non_lossless_reads_two_bit_tx_mode_without_select() {
        // Reads tx_mode = ALLOW_16X16 (=2); since it's not ALLOW_32X32, no extra bit is read.
        // maxTxSize = TX_16X16, so read_coef_probs runs 3 times for TX_4X4, TX_8X8, TX_16X16.
        let mut enc = BoolEncoder::new();
        enc.write_literal(2, 2); // tx_mode = ALLOW_16X16
        enc.write_literal(0, 1); // txSz=TX_4X4 update_probs=0
        enc.write_literal(0, 1); // txSz=TX_8X8 update_probs=0
        enc.write_literal(0, 1); // txSz=TX_16X16 update_probs=0
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, false).expect("should parse");
        assert_eq!(header.tx_mode, 2); // ALLOW_16X16
        assert_eq!(header.probs, CompressedHeaderProbs::default());
    }

    #[test]
    fn tx_mode_select_reads_tx_mode_probs_and_full_coef_range() {
        // tx_mode = ALLOW_32X32 (=3) + tx_mode_select(1) = TX_MODE_SELECT (=4)
        let mut enc = BoolEncoder::new();
        enc.write_literal(3, 2); // tx_mode raw = ALLOW_32X32
        enc.write_literal(1, 1); // tx_mode_select = 1 -> tx_mode = TX_MODE_SELECT
                                 // tx_mode_probs(): 8x8(2*1) + 16x16(2*2) + 32x32(2*3) = 12 calls to diff_update_prob
        for _ in 0..12 {
            enc.write_bool(false, 252);
        }
        // read_coef_probs: maxTxSize = TX_32X32, so txSz = 0..=3, 4 iterations
        for _ in 0..4 {
            enc.write_literal(0, 1);
        }
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        enc.write_bool(false, 252);
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, false).expect("should parse");
        assert_eq!(header.tx_mode, TX_MODE_SELECT);
        assert_eq!(header.probs, CompressedHeaderProbs::default());
    }

    #[test]
    fn diff_update_prob_actually_changes_skip_prob() {
        // Updates skip_prob[0]: update_prob=1, decode_term_subexp()'s first
        // branch (bit=0 -> sub_exp_val (L(4))) yields deltaProb=5, and inv_remap_prob(5, 192) is applied.
        let mut enc = BoolEncoder::new();
        enc.write_literal(0, 1); // read_coef_probs: txSz=TX_4X4, update_probs=0
        enc.write_bool(true, 252); // skip_prob[0]: update_prob=1
        enc.write_literal(0, 1); // decode_term_subexp: bit=0
        enc.write_literal(5, 4); // sub_exp_val = 5
        enc.write_bool(false, 252); // skip_prob[1]: update_prob=0
        enc.write_bool(false, 252); // skip_prob[2]: update_prob=0
        let buf = enc.finish();

        let header = parse_compressed_header(&buf, true).expect("should parse");
        let expected = inv_remap_prob(5, DEFAULT_SKIP_PROB[0]);
        assert_eq!(header.probs.skip_prob[0], expected);
        assert_ne!(header.probs.skip_prob[0], DEFAULT_SKIP_PROB[0]);
        assert_eq!(header.probs.skip_prob[1], DEFAULT_SKIP_PROB[1]);
        assert_eq!(header.probs.skip_prob[2], DEFAULT_SKIP_PROB[2]);
    }

    #[test]
    fn inv_recenter_nonneg_matches_spec_cases() {
        // v > 2m -> returns v as-is
        assert_eq!(inv_recenter_nonneg(10, 3), 10);
        // v is odd -> m - (v+1)/2
        assert_eq!(inv_recenter_nonneg(3, 5), 5 - 2);
        // v is even -> m + v/2
        assert_eq!(inv_recenter_nonneg(4, 5), 5 + 2);
    }

    #[test]
    fn empty_data_is_rejected() {
        let data: [u8; 0] = [];
        assert_eq!(
            parse_compressed_header(&data, true).unwrap_err(),
            CompressedHeaderError::BoolCoder(BoolCoderError::EmptyBuffer)
        );
    }
}
