//! vp9dec: a from-scratch VP9 video decoder (zero dependency crates).
//!
//! Reference spec: VP9 Bitstream & Decoding Process Specification v0.7
//! (Google, dated 2017-02-22, <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>)
//!
//! # Milestones
//! - M1: IVF container parser, bool decoder, uncompressed frame header parsing
//! - M2: keyframe decoding via intra prediction
//! - M2b: loop filter (deblocking filter) + official conformance verification
//! - M3 first half: inter frame bitstream decoding (up to but not including motion compensation)
//! - M3 second half: motion compensation, probability adaptation, reference frame management + full-frame MD5 conformance
//! - M4: full conformance test pass
//!
//! (Modules are added incrementally in subsequent commits.)

pub mod bit_reader;
pub mod bool_coder;
pub mod compressed_header;
pub mod counts;
pub mod dpb;
pub mod framebuffer;
pub mod header;
pub mod ivf;
pub mod loop_filter;
pub mod md5;
pub mod mv;
pub mod predict;
pub mod prob_tables;
pub mod quant;
pub mod scan;
pub mod superframe;
pub mod tile;
pub mod transform;

use compressed_header::{
    parse_compressed_header_ex, CompressedHeaderError, FrameContext, FrameContextStore,
};
use counts::{adapt_coef_probs, adapt_noncoef_probs};
use dpb::{Dpb, RefFrameData};
use header::{
    parse_uncompressed_header, ColorConfig, FrameHeader, FrameType, HeaderError, NUM_REF_FRAMES,
};
use tile::{MiGrid, TileDecoder, TileError};

/// Error type covering everything that can cause [`decode_keyframe`]/[`Decoder::decode_frame`] to fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Failed to parse the uncompressed header (`uncompressed_header`).
    Header(HeaderError),
    /// Failed to parse the compressed header (`compressed_header`).
    CompressedHeader(CompressedHeaderError),
    /// Failed to decode tiles, mode info, or tokens.
    Tile(TileError),
    /// [`decode_keyframe`] was passed a non-keyframe (inter frame or intra-only frame).
    NotAKeyFrame,
    /// Frame data is malformed, e.g. `header_size_in_bytes` exceeds the frame data length.
    TruncatedFrame,
    /// A frame that isn't 8-bit (`BitDepth == 8`). [`framebuffer::Plane`] is fixed to `u8`,
    /// so 10-bit/12-bit frames are currently unsupported.
    UnsupportedBitDepth(u8),
    /// The DPB slot referenced by `show_existing_frame` has no frame stored
    /// (does not occur with normal conformance bitstreams).
    MissingReferenceFrame,
}

impl From<HeaderError> for DecodeError {
    fn from(e: HeaderError) -> Self {
        DecodeError::Header(e)
    }
}

impl From<CompressedHeaderError> for DecodeError {
    fn from(e: CompressedHeaderError) -> Self {
        DecodeError::CompressedHeader(e)
    }
}

impl From<TileError> for DecodeError {
    fn from(e: TileError) -> Self {
        DecodeError::Tile(e)
    }
}

/// One decoded frame. Cropped to the display size (`FrameWidth`/`FrameHeight`),
/// holding the 3 YUV420 planes as row-major `Vec<u8>`.
///
/// The `u`/`v` sizes follow the output process of spec §8.9, computed as
/// `((width + subsampling_x) >> subsampling_x) x ((height + subsampling_y) >> subsampling_y)`
/// (for 4:2:0 this is `((width+1)/2) x ((height+1)/2)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Crops the frame buffer (`planes`) to the display size and builds a [`Frame`].
fn crop_to_frame(
    planes: &[framebuffer::Plane; 3],
    width: u32,
    height: u32,
    color_config: &ColorConfig,
) -> Frame {
    let sub_x = color_config.subsampling_x as u32;
    let sub_y = color_config.subsampling_y as u32;
    let uv_width = ((width + sub_x) >> sub_x) as usize;
    let uv_height = ((height + sub_y) >> sub_y) as usize;

    Frame {
        width,
        height,
        y: planes[0].crop(width as usize, height as usize),
        u: planes[1].crop(uv_width, uv_height),
        v: planes[2].crop(uv_width, uv_height),
    }
}

/// Builds a [`RefFrameData`] for DPB storage using the same crop calculation as
/// [`crop_to_frame`] (spec §8.10 "Reference frame update process" step 1).
fn build_ref_frame_data(
    planes: &[framebuffer::Plane; 3],
    width: u32,
    height: u32,
    color_config: &ColorConfig,
) -> RefFrameData {
    let sub_x = color_config.subsampling_x as u32;
    let sub_y = color_config.subsampling_y as u32;
    let uv_width = ((width + sub_x) >> sub_x) as usize;
    let uv_height = ((height + sub_y) >> sub_y) as usize;

    RefFrameData {
        width,
        height,
        subsampling_x: sub_x,
        subsampling_y: sub_y,
        bit_depth: color_config.bit_depth,
        y: planes[0].crop_to_plane(width as usize, height as usize),
        u: planes[1].crop_to_plane(uv_width, uv_height),
        v: planes[2].crop_to_plane(uv_width, uv_height),
    }
}

/// Read-only per-frame decode statistics, recorded purely for observation (e.g. test
/// assertions that a stream actually exercised a given decode path). Has no effect on
/// decode behavior. See [`Decoder::last_frame_info`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDecodeInfo {
    /// `intra_only` (spec §6.2). `false` for key frames.
    pub intra_only: bool,
    /// `FrameIsIntra` (spec §6.2). `true` for key frames and intra-only frames.
    pub frame_is_intra: bool,
    /// `reset_frame_context` (spec §7.2). Always 0 for key frames and when
    /// `error_resilient_mode == 1`.
    pub reset_frame_context: u8,
    /// `segmentation_enabled` (spec §6.2.11).
    pub segmentation_enabled: bool,
    /// Indexed by `header::SEG_LVL_*`. `seg_features_active[level]` is `true` if
    /// `segmentation_enabled` and `FeatureEnabled[segment][level]` is set for any of
    /// the `MAX_SEGMENTS` segments in this frame.
    pub seg_features_active: [bool; header::SEG_LVL_MAX],
}

fn ref_frame_data_to_frame(data: &RefFrameData) -> Frame {
    Frame {
        width: data.width,
        height: data.height,
        y: data.y.crop(data.y.width, data.y.height),
        u: data.u.crop(data.u.width, data.u.height),
        v: data.v.crop(data.v.width, data.v.height),
    }
}

/// Decodes a single keyframe.
///
/// `frame_data` is one frame's worth of VP9 bitstream extracted from an IVF or similar
/// container (raw data starting at `uncompressed_header`). Returns an error if
/// `show_existing_frame == 1` or `frame_type != KEY_FRAME`.
///
/// A keyframe references no other frames and always starts probability tables from
/// their defaults, so this is provided as a standalone function that carries no
/// inter-frame state ([`Decoder`]) (internally it calls [`Decoder::decode_frame`]
/// via a disposable [`Decoder`]). To read through multiple frames in sequence
/// (including inter frames), use [`Decoder`] directly.
///
/// After tile decoding and before cropping, the loop filter (spec §8.8, [`crate::loop_filter`]) is applied.
pub fn decode_keyframe(frame_data: &[u8]) -> Result<Frame, DecodeError> {
    // Confirm up front that this is a keyframe (decode_frame also accepts non-keyframes).
    let dummy_ref_sizes = [(0u32, 0u32); NUM_REF_FRAMES];
    let dummy_lf_deltas = (
        header::DEFAULT_LOOP_FILTER_REF_DELTAS,
        header::DEFAULT_LOOP_FILTER_MODE_DELTAS,
    );
    let dummy_seg_features = (
        [[false; header::SEG_LVL_MAX]; header::MAX_SEGMENTS],
        [[0i32; header::SEG_LVL_MAX]; header::MAX_SEGMENTS],
        false,
    );
    let (parsed, _consumed) = parse_uncompressed_header(
        frame_data,
        &dummy_ref_sizes,
        dummy_lf_deltas,
        dummy_seg_features,
    )?;
    match &parsed {
        FrameHeader::New(h) if h.frame_type == FrameType::KeyFrame => {}
        FrameHeader::New(_) => return Err(DecodeError::NotAKeyFrame),
        FrameHeader::ShowExistingFrame { .. } => return Err(DecodeError::NotAKeyFrame),
    }

    let mut decoder = Decoder::new();
    match decoder.decode_frame(frame_data)? {
        Some(frame) => Ok(frame),
        None => unreachable!("a keyframe always has show_frame == 1 (required by frame_is_intra)"),
    }
}

/// A stateful decoder for decoding multiple frames in sequence.
///
/// VP9 carries the following state across frames, so frames cannot be processed
/// independently one by one (`decode_keyframe` handles only standalone keyframes,
/// so it carries none of this state):
/// - Reference frame slots (`RefFrameWidth`/`RefFrameHeight`, spec §6.2.5
///   `frame_size_with_refs`) and the actual pixel data ([`Dpb`], spec §8.10).
/// - Frame contexts (the 4 probability table slots selected by `frame_context_idx`,
///   spec §7.1.2 `load_probs`/`save_probs`). See [`FrameContextStore`].
/// - The previous frame's `Mvs`/`RefFrames`, referenced when `UsePrevFrameMvs`
///   (spec §7.2.6) is true (this implementation keeps the previous frame's
///   [`MiGrid`] in its entirety).
/// - The loop filter's `ref_deltas`/`mode_deltas` (spec §7.2, reset only by
///   `setup_past_independence()`).
/// - `LastFrameType` (spec §7.2; used in the `updateFactor` calculation for
///   probability adaptation, spec §8.4.3).
pub struct Decoder {
    ref_frame_sizes: [(u32, u32); NUM_REF_FRAMES],
    frame_contexts: FrameContextStore,
    /// Used for computing `UsePrevFrameMvs` (spec §7.2.6). Not updated on
    /// `show_existing_frame` (because `compute_image_size` isn't called).
    prev_frame_dims: Option<(u32, u32)>,
    prev_show_frame: Option<bool>,
    /// The previous frame's `Mvs`/`RefFrames` (equivalent to `PrevMvs`/`PrevRefFrames`).
    prev_mi_grid: Option<MiGrid>,
    /// Inter frames and intra-only frames (`Profile == 0`) don't resend
    /// `color_config` in the bitstream, so the value is carried over from the
    /// most recent keyframe/intra-only frame (spec §7.2: it's a conformance
    /// requirement that bit depth and subsampling match the reference frame).
    last_color_config: Option<ColorConfig>,
    /// The actual pixel data of reference frames (spec §8.10 `FrameStore`).
    dpb: Dpb,
    /// The loop filter's `ref_deltas`/`mode_deltas` (state that persists per spec §7.2).
    loop_filter_deltas: ([i8; 4], [i8; 2]),
    /// The segmentation `FeatureEnabled`/`FeatureData`/`segmentation_abs_or_delta_update`
    /// state that persists across frames per spec §7.2.10 (reset by `setup_past_independence()`).
    segmentation_features: (
        [[bool; header::SEG_LVL_MAX]; header::MAX_SEGMENTS],
        [[i32; header::SEG_LVL_MAX]; header::MAX_SEGMENTS],
        bool,
    ),
    /// `PrevSegmentIds` (spec §6.4.14), row-major `MiRows x MiCols` (unpadded).
    /// Cleared to all-zero on the first frame and whenever the frame size changes
    /// (spec §7.2.6), updated after decoding only when
    /// `segmentation_enabled && segmentation_update_map` (spec §8.1 step 3).
    prev_segment_ids: Vec<u8>,
    /// `LastFrameType` (spec §7.2). Not updated on `show_existing_frame` frames.
    last_frame_type: Option<FrameType>,
    /// Observation-only stats from the most recently decoded frame (see
    /// [`FrameDecodeInfo`]). Not updated on `show_existing_frame` frames (there is no
    /// new uncompressed header to read the values from).
    last_frame_info: Option<FrameDecodeInfo>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            ref_frame_sizes: [(0, 0); NUM_REF_FRAMES],
            frame_contexts: FrameContextStore::new(),
            prev_frame_dims: None,
            prev_show_frame: None,
            prev_mi_grid: None,
            last_color_config: None,
            dpb: Dpb::new(),
            loop_filter_deltas: (
                header::DEFAULT_LOOP_FILTER_REF_DELTAS,
                header::DEFAULT_LOOP_FILTER_MODE_DELTAS,
            ),
            segmentation_features: (
                [[false; header::SEG_LVL_MAX]; header::MAX_SEGMENTS],
                [[0; header::SEG_LVL_MAX]; header::MAX_SEGMENTS],
                false,
            ),
            prev_segment_ids: Vec::new(),
            last_frame_type: None,
            last_frame_info: None,
        }
    }

    /// Observation-only stats from the most recently decoded frame, for tests to
    /// confirm a stream actually exercised a given decode path. `None` before the
    /// first call to [`Decoder::decode_frame`] (and unchanged by
    /// `show_existing_frame` frames, which parse no new uncompressed header).
    pub fn last_frame_info(&self) -> Option<FrameDecodeInfo> {
        self.last_frame_info
    }

    /// Resets `PrevSegmentIds` to all-zero when the spec requires it, before tile decode
    /// (an error-resilient inter frame reads the map during its own decode):
    /// - `compute_image_size()` step 1 (spec §7.2.6): first invocation or
    ///   FrameWidth/FrameHeight changed since the previous invocation (detected via
    ///   `prev_frame_dims`, read before `decode_frame` overwrites it at the end).
    /// - `setup_past_independence()` (spec §7.2): `FrameIsIntra || error_resilient_mode`
    ///   (the caller passes this as `setup_past_independence`).
    fn clear_prev_segment_ids_if_needed(
        &mut self,
        dims: (u32, u32),
        image_size: &header::ImageSize,
        setup_past_independence: bool,
    ) {
        if setup_past_independence || self.prev_frame_dims != Some(dims) {
            self.prev_segment_ids = vec![0u8; (image_size.mi_cols * image_size.mi_rows) as usize];
        }
    }

    /// Decodes one container chunk (one IVF frame, one WebM block, etc.). Callers must pass
    /// chunks extracted from an IVF or similar container in bitstream order (decode order),
    /// not display order.
    ///
    /// A chunk may pack more than one VP9 frame via the "superframe" mechanism -- see
    /// [`superframe`] -- so this splits it with [`superframe::split_superframe`] and decodes
    /// each contained VP9 frame in turn through [`Decoder::decode_one_frame`].
    /// `show_existing_frame` chunks (which never carry a superframe index) pass through as
    /// their own single frame.
    ///
    /// The return value indicates whether the chunk produced a displayable frame: `Some(Frame)`
    /// if one of its constituent frames has `show_existing_frame == 1` or `show_frame == 1`
    /// (a VP9 superframe has at most one such frame), otherwise (every constituent frame is
    /// hidden, i.e. a droppable/altref frame) `None`. Internal state (reference frame buffers,
    /// frame context, previous frame's MVs, etc.) is updated correctly in either case, and
    /// [`Decoder::last_frame_info`] reflects the last constituent frame decoded.
    pub fn decode_frame(&mut self, chunk: &[u8]) -> Result<Option<Frame>, DecodeError> {
        let mut displayed = None;
        for frame_data in superframe::split_superframe(chunk) {
            if let Some(frame) = self.decode_one_frame(frame_data)? {
                displayed = Some(frame);
            }
        }
        Ok(displayed)
    }

    /// Decodes exactly one VP9 frame (not a container chunk -- see [`Decoder::decode_frame`],
    /// the public entry point, which splits a chunk into these via [`superframe::split_superframe`]).
    ///
    /// The return value indicates whether this frame is displayable: `Some(Frame)` if
    /// `show_existing_frame == 1` or `show_frame == 1`, otherwise (a hidden frame with
    /// `show_frame == 0`, i.e. a droppable/altref frame) `None`. Internal state
    /// (reference frame buffers, frame context, previous frame's MVs, etc.) is updated
    /// correctly in either case.
    fn decode_one_frame(&mut self, frame_data: &[u8]) -> Result<Option<Frame>, DecodeError> {
        let (parsed, consumed) = parse_uncompressed_header(
            frame_data,
            &self.ref_frame_sizes,
            self.loop_filter_deltas,
            self.segmentation_features,
        )?;
        let mut header = match parsed {
            FrameHeader::New(h) => h,
            FrameHeader::ShowExistingFrame {
                frame_to_show_map_idx,
            } => {
                let data = self
                    .dpb
                    .get(frame_to_show_map_idx)
                    .ok_or(DecodeError::MissingReferenceFrame)?;
                return Ok(Some(ref_frame_data_to_frame(data)));
            }
        };

        if header.frame_is_intra {
            self.last_color_config = Some(header.color_config);
        } else if let Some(cc) = self.last_color_config {
            // Inter frames don't resend color_config (see comment in header.rs).
            header.color_config = cc;
        }

        if header.color_config.bit_depth != 8 {
            return Err(DecodeError::UnsupportedBitDepth(
                header.color_config.bit_depth,
            ));
        }

        let header_size = header.header_size_in_bytes as usize;
        let compressed_start = consumed;
        let compressed_end = compressed_start
            .checked_add(header_size)
            .ok_or(DecodeError::TruncatedFrame)?;
        if compressed_end > frame_data.len() {
            return Err(DecodeError::TruncatedFrame);
        }
        let compressed_bytes = &frame_data[compressed_start..compressed_end];

        // UsePrevFrameMvs (spec §7.2.6).
        let use_prev_frame_mvs = !header.frame_is_intra
            && !header.error_resilient_mode
            && self.prev_frame_dims == Some((header.width, header.height))
            && self.prev_show_frame == Some(true);

        let image_size = header::compute_image_size(header.width, header.height);
        self.clear_prev_segment_ids_if_needed(
            (header.width, header.height),
            &image_size,
            header.frame_is_intra || header.error_resilient_mode,
        );

        // setup_past_independence() (spec §7.2): called only when FrameIsIntra ||
        // error_resilient_mode. Its `save_probs` calls reset 0, 1, or all 4 stored
        // frame context slots to defaults depending on reset_frame_context (see
        // `frame_context_reset`). frame_context_idx itself is pinned to 0 for the
        // subsequent load/save (already corrected on the header.rs side).
        if header.frame_is_intra || header.error_resilient_mode {
            match frame_context_reset(
                header.frame_type,
                header.error_resilient_mode,
                header.reset_frame_context,
                header.frame_context_idx_raw,
            ) {
                FrameContextReset::All => self.frame_contexts.reset_all(),
                FrameContextReset::Slot(idx) => {
                    self.frame_contexts.save(idx, FrameContext::default())
                }
                FrameContextReset::None => {}
            }
        }
        // The starting value equivalent to `load_probs`/`load_probs2` (spec §6.1, start of
        // `frame()`). Kept around after the compressed_header call because `refresh_probs()`
        // (spec §6.1.2) restores it to this pre-forward-update value before applying
        // backward adaptation.
        let starting_probs = self.frame_contexts.load(header.frame_context_idx);

        let compressed = parse_compressed_header_ex(
            compressed_bytes,
            header.quantization.lossless,
            header.frame_is_intra,
            header.interpolation_filter,
            header.ref_frame_sign_bias,
            header.allow_high_precision_mv,
            starting_probs.clone(),
        )?;

        // Resolve the pixel data of the DPB slots this frame references, for motion
        // compensation (spec §8.5.2.3-8.5.2.4). Not referenced when `FrameIsIntra == 1`.
        let resolved_refs: [Option<RefFrameData>; 3] = if header.frame_is_intra {
            [None, None, None]
        } else {
            std::array::from_fn(|i| self.dpb.get(header.ref_frame_idx[i]).cloned())
        };

        let tile_data = &frame_data[compressed_end..];
        let prev_grid = if use_prev_frame_mvs {
            self.prev_mi_grid.clone()
        } else {
            None
        };
        let mut tile_decoder = TileDecoder::new_with_prev(
            &header,
            &compressed,
            use_prev_frame_mvs,
            prev_grid,
            resolved_refs,
            self.prev_segment_ids.clone(),
        );
        tile_decoder.decode_tiles(tile_data)?;
        tile_decoder.apply_loop_filter(&header.loop_filter);

        // spec §8.1 step 3: PrevSegmentIds is refreshed from this frame's SegmentIds only
        // when segmentation_enabled && segmentation_update_map (not gated by show_frame).
        // Otherwise it is left as-is (already reset to zero above if the size changed).
        if header.segmentation.enabled && header.segmentation.update_map {
            let grid = tile_decoder.mi_grid();
            let mut new_map = vec![0u8; (image_size.mi_cols * image_size.mi_rows) as usize];
            for row in 0..image_size.mi_rows {
                for col in 0..image_size.mi_cols {
                    new_map[(row * image_size.mi_cols + col) as usize] =
                        grid.get(row, col).segment_id;
                }
            }
            self.prev_segment_ids = new_map;
        }

        // refresh_probs() (spec §6.1.2).
        let final_probs = if !header.error_resilient_mode && !header.frame_parallel_decoding_mode {
            // load_probs( frame_context_idx ): restore all tables except tx_probs/skip_prob
            // to their pre-forward-update value (starting_probs). tx_probs/skip_prob are
            // left at their post-forward-update value from compressed_header().
            let mut working = starting_probs.clone();
            working.tx_probs = compressed.probs.tx_probs;
            working.skip_prob = compressed.probs.skip_prob;

            let counts = tile_decoder.counts();
            // Spec §8.4.3: determining updateFactor.
            let update_factor = if header.frame_is_intra {
                112
            } else if self.last_frame_type == Some(FrameType::KeyFrame) {
                128
            } else {
                112
            };
            adapt_coef_probs(&mut working.coef_probs, counts, update_factor);

            if !header.frame_is_intra {
                // load_probs2( frame_context_idx ): also restore tx_probs/skip_prob to
                // their pre-forward-update value before applying adapt_noncoef_probs.
                working.tx_probs = starting_probs.tx_probs;
                working.skip_prob = starting_probs.skip_prob;
                adapt_noncoef_probs(
                    &mut working,
                    counts,
                    header.interpolation_filter,
                    compressed.tx_mode,
                    header.allow_high_precision_mv,
                );
            }
            working
        } else {
            compressed.probs.clone()
        };
        if header.refresh_frame_context {
            self.frame_contexts
                .save(header.frame_context_idx, final_probs);
        }

        // Reference frame update process (spec §8.10).
        let ref_data = build_ref_frame_data(
            tile_decoder.planes(),
            header.width,
            header.height,
            &header.color_config,
        );
        self.dpb.update(header.refresh_frame_flags, &ref_data);
        for (slot, size) in self.ref_frame_sizes.iter_mut().enumerate() {
            if (header.refresh_frame_flags >> slot) & 1 == 1 {
                *size = (header.width, header.height);
            }
        }

        // compute_image_size is never called for show_existing_frame, so by the time we
        // reach here it has always been called (spec §7.2.6). Recorded for UsePrevFrameMvs.
        self.prev_frame_dims = Some((header.width, header.height));
        self.prev_show_frame = Some(header.show_frame);
        self.prev_mi_grid = Some(tile_decoder.mi_grid().clone());
        self.loop_filter_deltas = (
            header.loop_filter.ref_deltas,
            header.loop_filter.mode_deltas,
        );
        self.segmentation_features = (
            header.segmentation.feature_enabled,
            header.segmentation.feature_data,
            header.segmentation.abs_or_delta_update,
        );
        self.last_frame_type = Some(header.frame_type);
        self.last_frame_info = Some(FrameDecodeInfo {
            intra_only: header.intra_only,
            frame_is_intra: header.frame_is_intra,
            reset_frame_context: header.reset_frame_context,
            segmentation_enabled: header.segmentation.enabled,
            seg_features_active: std::array::from_fn(|level| {
                header.segmentation.enabled
                    && header.segmentation.feature_enabled.iter().any(|seg| seg[level])
            }),
        });

        if header.show_frame {
            Ok(Some(crop_to_frame(
                tile_decoder.planes(),
                header.width,
                header.height,
                &header.color_config,
            )))
        } else {
            Ok(None)
        }
    }
}

/// Which stored frame context slot(s) get reset to defaults, per the `save_probs`
/// calls inside `setup_past_independence()` (spec §7.2). Only meaningful when called
/// under `FrameIsIntra || error_resilient_mode` (the condition under which
/// `setup_past_independence()` runs at all; see the call site in `decode_frame`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameContextReset {
    /// `reset_frame_context` is 0 or 1: no stored context is touched.
    None,
    /// `reset_frame_context == 2`: only the slot at this (raw, pre-`setup_past_independence`) index is reset.
    Slot(u8),
    /// `frame_type == KEY_FRAME`, `error_resilient_mode`, or `reset_frame_context == 3`: all 4 slots are reset.
    All,
}

/// `frame_context_idx` must be the raw bitstream value (`NewFrameHeader::frame_context_idx_raw`),
/// since `save_probs( frame_context_idx )` runs before `setup_past_independence()` forces it to 0.
fn frame_context_reset(
    frame_type: FrameType,
    error_resilient_mode: bool,
    reset_frame_context: u8,
    frame_context_idx: u8,
) -> FrameContextReset {
    if frame_type == FrameType::KeyFrame || error_resilient_mode || reset_frame_context == 3 {
        FrameContextReset::All
    } else if reset_frame_context == 2 {
        FrameContextReset::Slot(frame_context_idx)
    } else {
        FrameContextReset::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_context_reset_keyframe_always_resets_all() {
        // frame_type == KEY_FRAME overrides reset_frame_context entirely.
        for reset_frame_context in 0..=3 {
            assert_eq!(
                frame_context_reset(FrameType::KeyFrame, false, reset_frame_context, 2),
                FrameContextReset::All
            );
        }
    }

    #[test]
    fn frame_context_reset_error_resilient_always_resets_all() {
        // error_resilient_mode overrides reset_frame_context entirely.
        for reset_frame_context in 0..=3 {
            assert_eq!(
                frame_context_reset(FrameType::NonKeyFrame, true, reset_frame_context, 2),
                FrameContextReset::All
            );
        }
    }

    #[test]
    fn frame_context_reset_intra_only_reset_frame_context_3_resets_all() {
        assert_eq!(
            frame_context_reset(FrameType::NonKeyFrame, false, 3, 2),
            FrameContextReset::All
        );
    }

    #[test]
    fn frame_context_reset_intra_only_reset_frame_context_2_resets_only_that_slot() {
        assert_eq!(
            frame_context_reset(FrameType::NonKeyFrame, false, 2, 0),
            FrameContextReset::Slot(0)
        );
        assert_eq!(
            frame_context_reset(FrameType::NonKeyFrame, false, 2, 3),
            FrameContextReset::Slot(3)
        );
    }

    #[test]
    fn frame_context_reset_intra_only_reset_frame_context_0_or_1_resets_nothing() {
        assert_eq!(
            frame_context_reset(FrameType::NonKeyFrame, false, 0, 1),
            FrameContextReset::None
        );
        assert_eq!(
            frame_context_reset(FrameType::NonKeyFrame, false, 1, 1),
            FrameContextReset::None
        );
    }

    /// `PrevSegmentIds` reset lifecycle (`clear_prev_segment_ids_if_needed`):
    /// zeroed by `setup_past_independence()` (spec §7.2) and by the first-frame /
    /// size-change condition of `compute_image_size()` (spec §7.2.6); retained for a
    /// same-size non-intra non-error-resilient frame.
    #[test]
    fn prev_segment_ids_reset_lifecycle() {
        // 16x16 -> MiCols = MiRows = 2 (4 entries).
        let dims = (16u32, 16u32);
        let image_size = header::compute_image_size(dims.0, dims.1);
        let seeded = || {
            let mut d = Decoder::new();
            d.prev_frame_dims = Some(dims);
            d.prev_segment_ids = vec![5u8; 4];
            d
        };

        // Same size, no setup_past_independence: the map is retained.
        let mut d = seeded();
        d.clear_prev_segment_ids_if_needed(dims, &image_size, false);
        assert_eq!(d.prev_segment_ids, vec![5u8; 4]);

        // setup_past_independence (FrameIsIntra || error_resilient_mode): zeroed even
        // though the size is unchanged.
        let mut d = seeded();
        d.clear_prev_segment_ids_if_needed(dims, &image_size, true);
        assert_eq!(d.prev_segment_ids, vec![0u8; 4]);

        // Size change (compute_image_size step 1): zeroed (and resized) even without
        // setup_past_independence.
        let mut d = seeded();
        let new_dims = (24u32, 16u32); // MiCols = 3, MiRows = 2 -> 6 entries.
        let new_image_size = header::compute_image_size(new_dims.0, new_dims.1);
        d.clear_prev_segment_ids_if_needed(new_dims, &new_image_size, false);
        assert_eq!(d.prev_segment_ids, vec![0u8; 6]);

        // First invocation (prev_frame_dims == None): zeroed.
        let mut d = Decoder::new();
        d.prev_segment_ids = vec![5u8; 4];
        d.clear_prev_segment_ids_if_needed(dims, &image_size, false);
        assert_eq!(d.prev_segment_ids, vec![0u8; 4]);
    }

    /// A `FrameContext` distinguishable from `FrameContext::default()`, standing in for a
    /// context that has been backward-adapted (spec §8.4) away from its default values.
    fn adapted_context() -> FrameContext {
        let mut ctx = FrameContext::default();
        ctx.mv_hp_prob = [1, 2];
        ctx
    }

    /// End-to-end check of `frame_context_reset`'s output against `FrameContextStore`:
    /// seeds all 4 slots with a non-default context, applies each reset outcome, and
    /// checks which slots came back to defaults vs. which retained the adapted value.
    #[test]
    fn frame_context_store_reset_application() {
        let seeded = || {
            let mut store = FrameContextStore::new();
            for i in 0..4 {
                store.save(i, adapted_context());
            }
            store
        };

        // None: every slot keeps the adapted context.
        let store = seeded();
        for i in 0..4 {
            assert_eq!(store.load(i), adapted_context());
        }

        // Slot(2): only slot 2 resets to defaults; the rest keep the adapted context.
        let mut store = seeded();
        if let FrameContextReset::Slot(idx) =
            frame_context_reset(FrameType::NonKeyFrame, false, 2, 2)
        {
            store.save(idx, FrameContext::default());
        } else {
            unreachable!();
        }
        for i in 0..4 {
            if i == 2 {
                assert_eq!(store.load(i), FrameContext::default());
            } else {
                assert_eq!(store.load(i), adapted_context());
            }
        }

        // All: every slot resets to defaults (e.g. keyframe).
        let mut store = seeded();
        assert_eq!(
            frame_context_reset(FrameType::KeyFrame, false, 0, 0),
            FrameContextReset::All
        );
        store.reset_all();
        for i in 0..4 {
            assert_eq!(store.load(i), FrameContext::default());
        }
    }

    /// `Decoder::last_frame_info()` (Part C observation surface): `None` before the
    /// first decode, then reflects the uncompressed header of the frame just decoded.
    /// Decodes the first (key) frame of an existing conformance vector rather than
    /// building a synthetic header, so this also exercises the real parse path.
    #[test]
    fn last_frame_info_reflects_a_decoded_keyframe() {
        let ivf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("vectors")
            .join("vp90-2-12-droppable_1.ivf");
        let ivf_bytes = match std::fs::read(&ivf_path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!(
                    "[skip] test vector not found, skipping: {}",
                    ivf_path.display()
                );
                return;
            }
        };

        let mut reader = ivf::IvfReader::new(&ivf_bytes).expect("failed to parse IVF header");
        let first_frame = reader
            .next()
            .expect("IVF file contains no frames")
            .expect("failed to read first IVF frame");

        let mut decoder = Decoder::new();
        assert_eq!(
            decoder.last_frame_info(),
            None,
            "last_frame_info() must be None before any frame is decoded"
        );

        decoder
            .decode_frame(first_frame.data)
            .expect("decode_frame failed on first frame");

        let info = decoder
            .last_frame_info()
            .expect("last_frame_info() must be Some after decoding a frame");
        // The first frame of any IVF stream is a key frame (spec §7.2 conformance requirement).
        assert!(info.frame_is_intra, "key frames have FrameIsIntra == true");
        assert!(!info.intra_only, "intra_only is only read for non-key frames");
        assert_eq!(
            info.reset_frame_context, 0,
            "reset_frame_context is only read for non-key, non-error-resilient frames"
        );
    }
}
