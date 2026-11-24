//! vp9dec: a from-scratch VP9 video decoder (zero dependency crates).
//!
//! Reference spec: VP9 Bitstream & Decoding Process Specification v0.7
//! (Google, dated 2017-02-22, <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>)
//!
//! Decodes all four VP9 profiles (8/10/12-bit; 4:2:0/4:2:2/4:4:0/4:4:4), verified bit-exact
//! against the full official conformance corpus. Runtime-detected AVX2 fast paths and
//! tile-parallel decoding accelerate the scalar reference path without altering its output.
//! See README.md "Current architecture" for the module map.

pub mod ivf;

// Modules below are crate-internal; a subset remain `#[doc(hidden)] pub mod` only so the
// pure-std integration tests in tests/ and examples/ can reach them -- not a stable API.
#[doc(hidden)]
pub mod bench_timing;
pub(crate) mod bit_reader;
#[doc(hidden)]
pub mod bool_coder;
pub(crate) mod common;
#[doc(hidden)]
pub mod compressed_header;
#[doc(hidden)]
pub mod counts;
#[doc(hidden)]
pub mod dpb;
#[doc(hidden)]
pub mod framebuffer;
#[doc(hidden)]
pub mod header;
#[doc(hidden)]
pub mod loop_filter;
pub(crate) mod mv_ref_tables;
#[doc(hidden)]
pub mod predict;
#[doc(hidden)]
pub mod prob_tables;
pub(crate) mod quant;
pub(crate) mod scan;
/// AVX2 SIMD kernels (inter prediction, intra prediction, loop filter, inverse transforms);
/// x86_64-only, see the hub `src/simd.rs` and its submodules.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod simd;
pub(crate) mod subpel;
#[doc(hidden)]
pub mod superframe;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub(crate) mod thread_pool;
#[doc(hidden)]
pub mod tile;
#[doc(hidden)]
pub mod transform;
#[cfg(test)]
#[path = "../tests/unit/support.rs"]
mod unit_test_support;

use std::fmt;
use std::sync::Arc;

use compressed_header::{
    parse_compressed_header, CompressedHeaderError, FrameContext, FrameContextStore,
};
use counts::{adapt_coef_probs, adapt_noncoef_probs};
use dpb::{Dpb, RefFrameData};
use header::{
    parse_uncompressed_header, ColorConfig, FrameHeader, FrameType, HeaderError, LoopFilterDeltas,
    PersistentState, SegFeaturePersist,
};
use tile::{MiGrid, TileDecoder, TileError};

/// Error type covering everything that can cause [`Decoder::decode_frame`] to fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Failed to parse the uncompressed header (`uncompressed_header`). Two variants are
    /// raised by the [`Decoder`] itself rather than the parser: `FrameSizeTooLarge` (a
    /// signaled size the decoder refuses to allocate) and `RefFrameFormatMismatch` (an
    /// inter frame whose reference doesn't share its bit depth / subsampling).
    Header(HeaderError),
    /// Failed to parse the compressed header (`compressed_header`).
    CompressedHeader(CompressedHeaderError),
    /// Failed to decode tiles, mode info, or tokens.
    Tile(TileError),
    /// Frame data is malformed, e.g. `header_size_in_bytes` exceeds the frame data length.
    TruncatedFrame,
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

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Header(e) => write!(f, "uncompressed header error: {e}"),
            DecodeError::CompressedHeader(e) => write!(f, "compressed header error: {e}"),
            DecodeError::Tile(e) => write!(f, "tile decode error: {e}"),
            DecodeError::TruncatedFrame => f.write_str("truncated VP9 frame data"),
            DecodeError::MissingReferenceFrame => {
                f.write_str("show_existing_frame references an empty DPB slot")
            }
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::Header(e) => Some(e),
            DecodeError::CompressedHeader(e) => Some(e),
            DecodeError::Tile(e) => Some(e),
            DecodeError::TruncatedFrame | DecodeError::MissingReferenceFrame => None,
        }
    }
}

/// One decoded plane's pixel data. 8-bit streams (`BitDepth == 8`) decode into `U8` (no
/// memory bloat from widening the overwhelmingly common case); 10/12-bit streams (VP9
/// profiles 2/3) decode into `U16` (each sample 0..=1023 or 0..=4095).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaneData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl PlaneData {
    /// Number of samples (not bytes) -- e.g. `width * height` for a cropped plane, the same
    /// meaning regardless of variant.
    pub fn len(&self) -> usize {
        match self {
            PlaneData::U8(v) => v.len(),
            PlaneData::U16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows the samples as 8-bit. Panics if this plane is `U16` (10/12-bit) -- for
    /// consumers that only ever handle 8-bit output.
    pub fn as_u8(&self) -> &[u8] {
        match self {
            PlaneData::U8(v) => v,
            PlaneData::U16(_) => panic!("PlaneData::as_u8 called on a 16-bit (10/12-bit) plane"),
        }
    }
}

/// One decoded frame. Cropped to the display size (`FrameWidth`/`FrameHeight`), holding the
/// 3 Y/U/V planes as [`PlaneData`] (row-major). Chroma sizing follows
/// `subsampling_x`/`subsampling_y`: 4:2:0 for profiles 0/2; profiles 1/3 also output
/// 4:2:2, 4:4:0, or 4:4:4.
///
/// The `u`/`v` sizes follow the output process of spec §8.9, computed as
/// `((width + subsampling_x) >> subsampling_x) x ((height + subsampling_y) >> subsampling_y)`
/// (for 4:2:0 this is `((width+1)/2) x ((height+1)/2)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub subsampling_x: u32,
    pub subsampling_y: u32,
    pub y: PlaneData,
    pub u: PlaneData,
    pub v: PlaneData,
}

/// Crops one plane to `(w, h)` and narrows to `PlaneData::U8`/`U16` depending on `bit_depth`
/// (8-bit output stays byte-sized; 10/12-bit needs the full `u16` range).
fn plane_to_plane_data(plane: &framebuffer::Plane, w: usize, h: usize, bit_depth: u8) -> PlaneData {
    if bit_depth == 8 {
        PlaneData::U8(plane.crop_u8(w, h))
    } else {
        PlaneData::U16(plane.crop(w, h))
    }
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
    let bit_depth = color_config.bit_depth;

    Frame {
        width,
        height,
        bit_depth,
        subsampling_x: sub_x,
        subsampling_y: sub_y,
        y: plane_to_plane_data(&planes[0], width as usize, height as usize, bit_depth),
        u: plane_to_plane_data(&planes[1], uv_width, uv_height, bit_depth),
        v: plane_to_plane_data(&planes[2], uv_width, uv_height, bit_depth),
    }
}

/// Builds a 10/12-bit [`RefFrameData`] for DPB storage by taking ownership of the decoded `planes`
/// and cropping their logical dimensions in-place (zero allocation, zero copy; the backing buffers
/// keep their superblock-aligned stride and are recycled when evicted from the DPB).
fn build_ref_frame_data_u16_in_place(
    mut planes: [framebuffer::Plane; 3],
    width: u32,
    height: u32,
    color_config: &ColorConfig,
) -> RefFrameData {
    let sub_x = color_config.subsampling_x as u32;
    let sub_y = color_config.subsampling_y as u32;
    let uv_width = ((width + sub_x) >> sub_x) as usize;
    let uv_height = ((height + sub_y) >> sub_y) as usize;

    planes[0].crop_in_place(width as usize, height as usize);
    planes[1].crop_in_place(uv_width, uv_height);
    planes[2].crop_in_place(uv_width, uv_height);
    let [y, u, v] = planes;

    RefFrameData {
        width,
        height,
        subsampling_x: sub_x,
        subsampling_y: sub_y,
        bit_depth: color_config.bit_depth,
        y: framebuffer::RefPlane::from_u16_plane(y),
        u: framebuffer::RefPlane::from_u16_plane(u),
        v: framebuffer::RefPlane::from_u16_plane(v),
    }
}

/// Read-only per-frame decode statistics, recorded purely for observation (e.g. test
/// assertions that a stream actually exercised a given decode path). Has no effect on
/// decode behavior. See [`DecodedFrame::info`].
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
        bit_depth: data.bit_depth,
        subsampling_x: data.subsampling_x,
        subsampling_y: data.subsampling_y,
        y: data.y.to_plane_data(),
        u: data.u.to_plane_data(),
        v: data.v.to_plane_data(),
    }
}

/// The outcome of decoding one constituent VP9 frame of a container chunk; one element
/// of [`Decoder::decode_frame`]'s result.
///
/// The two fields are `Option` independently because the three kinds of constituent
/// frame populate them differently:
/// - a shown decoded frame (`show_frame == 1`): `info: Some`, `frame: Some`
/// - a hidden decoded frame (`show_frame == 0`, e.g. an altref): `info: Some`, `frame: None`
/// - `show_existing_frame == 1`: `info: None` (no uncompressed header is parsed, so
///   there are no stats to report), `frame: Some` (the referenced DPB slot's pixels)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    /// Decode statistics, for observation only (see [`FrameDecodeInfo`]).
    pub info: Option<FrameDecodeInfo>,
    /// The displayable picture, cropped to display size. `None` for hidden frames.
    pub frame: Option<Frame>,
}

/// A stateful decoder for decoding multiple frames in sequence.
///
/// VP9 carries the following state across frames, so frames cannot be processed
/// independently one by one:
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
    /// Cross-frame state spec §7.2 requires: `RefFrameWidth`/`RefFrameHeight`, the loop
    /// filter's `ref_deltas`/`mode_deltas`, and the segmentation feature state -- see
    /// [`PersistentState`].
    persist: PersistentState,
    frame_contexts: Box<FrameContextStore>,
    /// Used for computing `UsePrevFrameMvs` (spec §7.2.6). Not updated on
    /// `show_existing_frame` (because `compute_image_size` isn't called).
    prev_frame_dims: Option<(u32, u32)>,
    prev_show_frame: Option<bool>,
    /// The previous frame's `Mvs`/`RefFrames` (equivalent to `PrevMvs`/`PrevRefFrames`).
    /// `Arc`-wrapped so handing a read-only copy to next frame's `TileDecoder` (when
    /// `UsePrevFrameMvs`) is a refcount bump rather than a deep clone.
    prev_mi_grid: Option<Arc<MiGrid>>,
    /// Inter frames and intra-only frames (`Profile == 0`) don't resend
    /// `color_config` in the bitstream, so the value is carried over from the
    /// most recent keyframe/intra-only frame (spec §7.2: it's a conformance
    /// requirement that bit depth and subsampling match the reference frame).
    last_color_config: Option<ColorConfig>,
    /// The actual pixel data of reference frames (spec §8.10 `FrameStore`).
    dpb: Dpb,
    /// `PrevSegmentIds` (spec §6.4.14), row-major `MiRows x MiCols` (unpadded).
    /// Cleared to all-zero on the first frame and whenever the frame size changes
    /// (spec §7.2.6), updated after decoding only when
    /// `segmentation_enabled && segmentation_update_map` (spec §8.1 step 3). `Arc`-wrapped
    /// so the per-frame clone-in to `TileDecoder` (unconditional, unlike `prev_mi_grid`) is
    /// a refcount bump rather than a deep clone.
    prev_segment_ids: Arc<Vec<u8>>,
    /// `LastFrameType` (spec §7.2). Not updated on `show_existing_frame` frames.
    last_frame_type: Option<FrameType>,
    /// Recycled whole-frame `[Plane; 3]` working buffers (and evicted 10/12-bit DPB planes),
    /// at most one per active resolution geometry.
    plane_pool: Vec<[framebuffer::Plane; 3]>,
    /// Recycled cropped `[Vec<u8>; 3]` buffers from evicted 8-bit DPB slots (at most one per
    /// active resolution geometry).
    ref_u8_pool: Vec<[Vec<u8>; 3]>,
    /// Recycled `MiGrid` allocation from the replaced `prev_mi_grid`.
    mi_grid_pool: Option<MiGrid>,
    /// Recycled frame-width above contexts for `TileDecoder`.
    above_ctx_pool: Option<tile::AboveContexts>,
    /// Recycled heap-boxed per-block scratch for `TileDecoder`.
    tile_scratch_pool: Option<Box<tile::TileScratch>>,
    /// Recycled syntax counter table for frames with backward adaptation enabled.
    counts_pool: Option<Box<counts::Counts>>,
    /// Persistent pure-`std` worker pool shared between tile-parallel decode and wavefront loop
    /// filtering (lazily spawned on first parallel frame, joined cleanly on drop).
    thread_pool: thread_pool::ThreadPool,
    /// Pooled per-tile-column worker `TileDecoder`s (reusing their column-strip `Plane`/`MiGrid`
    /// allocations across multi-tile frames).
    tile_worker_pool: Vec<TileDecoder>,
}

/// Cap on pooled frame-buffer sets held by [`Decoder`] (at most one per active geometry,
/// covering up to 3-layer spatial SVC without retaining stale resolutions).
const MAX_POOLED_PLANE_SETS: usize = 3;

type PlaneGeom = ((usize, usize), (usize, usize));

#[inline]
fn plane_set_geom(planes: &[framebuffer::Plane; 3]) -> PlaneGeom {
    (planes[0].backing_dims(), planes[1].backing_dims())
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            persist: PersistentState::default(),
            frame_contexts: Box::new(FrameContextStore::new()),
            prev_frame_dims: None,
            prev_show_frame: None,
            prev_mi_grid: None,
            last_color_config: None,
            dpb: Dpb::new(),
            prev_segment_ids: Arc::new(Vec::new()),
            last_frame_type: None,
            plane_pool: Vec::new(),
            ref_u8_pool: Vec::new(),
            mi_grid_pool: None,
            above_ctx_pool: None,
            tile_scratch_pool: None,
            counts_pool: None,
            thread_pool: thread_pool::ThreadPool::new(),
            tile_worker_pool: Vec::new(),
        }
    }

    /// Finds and removes a pooled `[Plane; 3]` reusable for `(y_w, y_h, uv_w, uv_h)`.
    /// Prefers an exact backing-dimension match, falling back to capacity reuse bounded by the
    /// largest live DPB resolution (so multi-layer 8-bit SVC shares one working buffer set with
    /// zero reallocations).
    fn take_matching_planes(
        &mut self,
        y_w: usize,
        y_h: usize,
        uv_w: usize,
        uv_h: usize,
    ) -> Option<[framebuffer::Plane; 3]> {
        let want: PlaneGeom = ((y_w, y_w * y_h), (uv_w, uv_w * uv_h));
        if let Some(idx) = self
            .plane_pool
            .iter()
            .position(|p| plane_set_geom(p) == want)
        {
            return Some(self.plane_pool.swap_remove(idx));
        }
        let (max_y, max_uv) = self.dpb.max_sb_plane_elems();
        let max_y = max_y.max(y_w * y_h);
        let max_uv = max_uv.max(uv_w * uv_h);
        for idx in 0..self.plane_pool.len() {
            let p = &mut self.plane_pool[idx];
            if p[0].try_reuse_working(y_w, y_h, max_y)
                && p[1].try_reuse_working(uv_w, uv_h, max_uv)
                && p[2].try_reuse_working(uv_w, uv_h, max_uv)
            {
                return Some(self.plane_pool.swap_remove(idx));
            }
        }
        None
    }

    /// Finds and removes a pooled `[Vec<u8>; 3]` matching `(y_len, uv_len)`.
    fn take_matching_ref_u8(&mut self, y_len: usize, uv_len: usize) -> Option<[Vec<u8>; 3]> {
        let idx = self.ref_u8_pool.iter().position(|v| {
            v[0].capacity() >= y_len
                && v[1].capacity() >= uv_len
                && v[0].len() == y_len
                && v[1].len() == uv_len
        })?;
        Some(self.ref_u8_pool.swap_remove(idx))
    }

    /// Recycles `item` into `pool` (at most one entry per active geometry, pruning stale ones).
    fn recycle_by_geom<T, G: Copy + Eq>(
        pool: &mut Vec<T>,
        item: T,
        active_geom: G,
        geom_of: impl Fn(&T) -> G,
        in_dpb: impl Fn(G) -> bool,
    ) {
        let geom = geom_of(&item);
        pool.retain(|entry| {
            let g = geom_of(entry);
            g == active_geom || in_dpb(g)
        });
        if (geom == active_geom || in_dpb(geom))
            && pool.len() < MAX_POOLED_PLANE_SETS
            && !pool.iter().any(|entry| geom_of(entry) == geom)
        {
            pool.push(item);
        }
    }

    fn recycle_planes(&mut self, planes: [framebuffer::Plane; 3], active_geom: PlaneGeom) {
        let dpb = &self.dpb;
        Self::recycle_by_geom(
            &mut self.plane_pool,
            planes,
            active_geom,
            plane_set_geom,
            |g| dpb.has_u16_geometry(g),
        );
    }

    fn recycle_ref_u8(&mut self, bufs: [Vec<u8>; 3], active_u8_geom: (usize, usize)) {
        let dpb = &self.dpb;
        Self::recycle_by_geom(
            &mut self.ref_u8_pool,
            bufs,
            active_u8_geom,
            |b| (b[0].len(), b[1].len()),
            |g| dpb.has_u8_geometry(g),
        );
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
            self.prev_segment_ids = Arc::new(vec![
                0u8;
                (image_size.mi_cols * image_size.mi_rows)
                    as usize
            ]);
        }
    }

    /// Spec §8.1 step 3: `PrevSegmentIds` is refreshed from this frame's `SegmentIds` (read
    /// out of `grid`) only when segmentation_enabled && segmentation_update_map (not gated by
    /// show_frame). Otherwise it is left as-is (already reset to zero by
    /// [`Self::clear_prev_segment_ids_if_needed`] if the size changed).
    fn refresh_prev_segment_ids(
        &mut self,
        header: &header::NewFrameHeader,
        image_size: &header::ImageSize,
        grid: &MiGrid,
    ) {
        if header.segmentation.enabled && header.segmentation.update_map {
            let mut new_map = vec![0u8; (image_size.mi_cols * image_size.mi_rows) as usize];
            for row in 0..image_size.mi_rows {
                for col in 0..image_size.mi_cols {
                    new_map[(row * image_size.mi_cols + col) as usize] =
                        grid.get(row, col).segment_id();
                }
            }
            self.prev_segment_ids = Arc::new(new_map);
        }
    }

    /// Reference frame update process (spec §8.10) + optional shown-frame construction.
    ///
    /// - For **8-bit** frames (`bit_depth == 8`), narrows the cropped pixels once into packed
    ///   `u8` DPB planes (`RefPlaneData::U8`), immediately returns the working `u16` `[Plane; 3]`
    ///   back to `plane_pool`, and clones the compact `u8` planes if `show_frame` is set (so only
    ///   a single working `u16` buffer is ever needed and DPB slots use half the memory).
    /// - For **10/12-bit** frames, crops the `u16` planes in-place into the DPB with zero copy
    ///   and recycles evicted slots back into `plane_pool`.
    fn output_and_refresh_refs(
        &mut self,
        header: &header::NewFrameHeader,
        color_config: &ColorConfig,
        planes: [framebuffer::Plane; 3],
    ) -> Option<Frame> {
        let active_geom = plane_set_geom(&planes);
        let sub_x = color_config.subsampling_x as u32;
        let sub_y = color_config.subsampling_y as u32;
        let y_w = header.width as usize;
        let y_h = header.height as usize;
        let uv_w = ((header.width + sub_x) >> sub_x) as usize;
        let uv_h = ((header.height + sub_y) >> sub_y) as usize;

        if header.refresh_frame_flags == 0 {
            let frame = header
                .show_frame
                .then(|| crop_to_frame(&planes, header.width, header.height, color_config));
            self.recycle_planes(planes, active_geom);
            return frame;
        }

        let (frame, ref_data) = if color_config.bit_depth == 8 {
            let mut bufs = self
                .take_matching_ref_u8(y_w * y_h, uv_w * uv_h)
                .unwrap_or_else(|| [Vec::new(), Vec::new(), Vec::new()]);
            planes[0].narrow_u8_into(y_w, y_h, &mut bufs[0]);
            planes[1].narrow_u8_into(uv_w, uv_h, &mut bufs[1]);
            planes[2].narrow_u8_into(uv_w, uv_h, &mut bufs[2]);
            let [y_u8, u_u8, v_u8] = bufs;
            let frame = header.show_frame.then(|| Frame {
                width: header.width,
                height: header.height,
                bit_depth: 8,
                subsampling_x: sub_x,
                subsampling_y: sub_y,
                y: PlaneData::U8(y_u8.clone()),
                u: PlaneData::U8(u_u8.clone()),
                v: PlaneData::U8(v_u8.clone()),
            });
            self.recycle_planes(planes, active_geom);
            let ref_data = Arc::new(RefFrameData {
                width: header.width,
                height: header.height,
                subsampling_x: sub_x,
                subsampling_y: sub_y,
                bit_depth: 8,
                y: framebuffer::RefPlane::from_u8_cropped(y_w, y_h, y_u8),
                u: framebuffer::RefPlane::from_u8_cropped(uv_w, uv_h, u_u8),
                v: framebuffer::RefPlane::from_u8_cropped(uv_w, uv_h, v_u8),
            });
            (frame, ref_data)
        } else {
            let frame = header
                .show_frame
                .then(|| crop_to_frame(&planes, header.width, header.height, color_config));
            let ref_data = Arc::new(build_ref_frame_data_u16_in_place(
                planes,
                header.width,
                header.height,
                color_config,
            ));
            (frame, ref_data)
        };

        let active_u8_geom = (y_w * y_h, uv_w * uv_h);
        let mut evicted = Vec::new();
        self.dpb
            .update_and_recycle(header.refresh_frame_flags, &ref_data, |old| {
                evicted.push(old)
            });
        for old in evicted {
            match (
                old.y.into_backing(),
                old.u.into_backing(),
                old.v.into_backing(),
            ) {
                (Ok(y), Ok(u), Ok(v)) => self.recycle_ref_u8([y, u, v], active_u8_geom),
                (Err(y), Err(u), Err(v)) => self.recycle_planes([y, u, v], active_geom),
                _ => {}
            }
        }
        // Prune stale geometries and oversized working-plane capacities after the DPB update.
        let dpb = &self.dpb;
        let (max_y, max_uv) = dpb.max_sb_plane_elems();
        let max_y = max_y.max(active_geom.0 .1);
        let max_uv = max_uv.max(active_geom.1 .1);
        self.plane_pool.retain(|p| {
            let g = plane_set_geom(p);
            (g == active_geom || dpb.has_u16_geometry(g))
                && p[0].capacity() <= max_y
                && p[1].capacity() <= max_uv
                && p[2].capacity() <= max_uv
        });
        self.ref_u8_pool.retain(|b| {
            let g = (b[0].len(), b[1].len());
            g == active_u8_geom || dpb.has_u8_geometry(g)
        });
        for (slot, size) in self.persist.ref_frame_sizes.iter_mut().enumerate() {
            if (header.refresh_frame_flags >> slot) & 1 == 1 {
                *size = (header.width, header.height);
            }
        }
        frame
    }

    /// Decodes one container chunk (one IVF frame, one WebM block, etc.). Callers must pass
    /// chunks extracted from an IVF or similar container in bitstream order (decode order),
    /// not display order.
    ///
    /// A chunk may pack more than one VP9 frame via the "superframe" mechanism, so this
    /// splits it (`superframe::split_superframe`) and decodes each contained VP9 frame in
    /// turn, returning one [`DecodedFrame`] per constituent frame in bitstream order
    /// (`show_existing_frame` chunks, which never carry a superframe index, yield exactly
    /// one element). At most one element carries `frame: Some` -- when several
    /// constituents are marked shown (spatial-SVC streams mark every layer shown), only
    /// the last of them keeps its picture, matching libvpx's display behavior; the
    /// others still report their `info`. Internal state (reference frame buffers, frame
    /// context, previous frame's MVs, etc.) is updated for every constituent frame,
    /// shown or hidden.
    pub fn decode_frame(&mut self, chunk: &[u8]) -> Result<Vec<DecodedFrame>, DecodeError> {
        let mut decoded = Vec::new();
        for frame_data in superframe::split_superframe(chunk) {
            decoded.push(self.decode_one_frame(frame_data)?);
        }
        // Spec §5.26 permits a superframe to produce multiple output frames ("it is also legal
        // for a superframe to result in multiple output frames"), so §8.9's per-frame Output
        // process, read in isolation, would surface every constituent whose own show_frame is
        // 1. In practice this crate's conformance oracle (libvpx, via ffmpeg -- see
        // docs/implementation-notes.md "M4 wave 3") surfaces exactly one displayed frame per
        // chunk: each shown constituent overwrites a pending-output slot, so the LAST SHOWN
        // constituent survives (spatial-SVC streams mark every layer shown; only the top layer
        // is displayed). Mirror that: only the last constituent with a decoded picture keeps
        // its `frame`; earlier ones keep their `info` (decode stats) but have `frame` cleared.
        // Keying on last-SHOWN rather than positionally-last means a (conforming but unusual)
        // chunk ending in a hidden frame cannot lose its visible output. No-op for the common
        // case (at most one shown constituent per chunk).
        if let Some(last_shown) = decoded.iter().rposition(|df| df.frame.is_some()) {
            for df in &mut decoded[..last_shown] {
                df.frame = None;
            }
        }
        Ok(decoded)
    }

    /// Resolves this frame's effective [`ColorConfig`] and remembers it when `frame_is_intra`.
    ///
    /// `header.color_config` is `Some` on key frames and intra-only frames; inter frames inherit
    /// the most recent intra frame's config (falling back to a neutral 8-bit 4:2:0 placeholder if
    /// an inter frame arrives before any intra frame).
    fn resolve_color_config(&mut self, header: &header::NewFrameHeader) -> ColorConfig {
        let color_config = header.color_config.unwrap_or_else(|| {
            self.last_color_config
                .unwrap_or_else(ColorConfig::fallback_inter)
        });
        if header.frame_is_intra {
            self.last_color_config = Some(color_config);
        }
        color_config
    }

    /// Runs `setup_past_independence()`'s `save_probs` context resets (spec §7.2) when
    /// `FrameIsIntra || error_resilient_mode`.
    fn reset_frame_contexts_if_needed(&mut self, header: &header::NewFrameHeader) {
        if header.frame_is_intra || header.error_resilient_mode {
            match frame_context_reset(
                header.frame_type,
                header.error_resilient_mode,
                header.reset_frame_context,
                header.frame_context_idx,
            ) {
                FrameContextReset::All => self.frame_contexts.reset_all(),
                FrameContextReset::Slot(idx) => {
                    self.frame_contexts.save(idx, FrameContext::default())
                }
                FrameContextReset::None => {}
            }
        }
    }

    /// Resolves the 3 DPB reference slots for an inter frame (cheap `Arc` bumps) and validates
    /// that every present reference matches this frame's bit depth and chroma subsampling
    /// (spec §8.5.1 / libvpx `valid_ref_frame_img_fmt`).
    ///
    /// Reference *sizes* are deliberately validated at block use time in `decode_block` rather
    /// than here, because a conformant stream (e.g. 3-layer spatial SVC) may list an out-of-range
    /// slot it never predicts from.
    fn resolve_and_validate_refs(
        &self,
        header: &header::NewFrameHeader,
        color_config: &ColorConfig,
    ) -> Result<[Option<Arc<RefFrameData>>; 3], DecodeError> {
        let resolved_refs: [Option<Arc<RefFrameData>>; 3] = if header.frame_is_intra {
            [None, None, None]
        } else {
            std::array::from_fn(|i| self.dpb.get_arc(header.ref_frame_idx[i]))
        };
        for r in resolved_refs.iter().flatten() {
            if r.bit_depth != color_config.bit_depth
                || r.subsampling_x != color_config.subsampling_x as u32
                || r.subsampling_y != color_config.subsampling_y as u32
            {
                return Err(DecodeError::Header(HeaderError::RefFrameFormatMismatch));
            }
        }
        Ok(resolved_refs)
    }

    /// Updates cross-frame persistent state (dimensions, show flag, loop-filter deltas,
    /// segmentation feature persistence, and `LastFrameType`) after a newly decoded frame.
    fn update_persistent_state(&mut self, header: &header::NewFrameHeader) {
        // compute_image_size is never called for show_existing_frame, so by the time we
        // reach here it has always been called (spec §7.2.6). Recorded for UsePrevFrameMvs.
        self.prev_frame_dims = Some((header.width, header.height));
        self.prev_show_frame = Some(header.show_frame);
        self.persist.loop_filter_deltas = LoopFilterDeltas {
            ref_deltas: header.loop_filter.ref_deltas,
            mode_deltas: header.loop_filter.mode_deltas,
        };
        self.persist.segmentation = SegFeaturePersist {
            enabled: header.segmentation.feature_enabled,
            data: header.segmentation.feature_data,
            abs_or_delta: header.segmentation.abs_or_delta_update,
        };
        self.last_frame_type = Some(header.frame_type);
    }

    /// Stores `mi_grid` as `prev_mi_grid` and recycles the replaced allocation when unshared.
    fn store_and_recycle_mi_grid(&mut self, mut mi_grid: MiGrid) {
        mi_grid.clear_sub_mvs();
        if let Some(old_grid) = self.prev_mi_grid.replace(Arc::new(mi_grid)) {
            if let Ok(owned) = Arc::try_unwrap(old_grid) {
                self.mi_grid_pool = Some(owned);
            }
        }
    }

    /// Decodes exactly one VP9 frame (not a container chunk -- see [`Decoder::decode_frame`],
    /// the public entry point, which splits a chunk into these via
    /// `superframe::split_superframe`).
    ///
    /// See [`DecodedFrame`] for how `info`/`frame` reflect the three kinds of constituent
    /// frame. Internal state (reference frame buffers, frame context, previous frame's
    /// MVs, etc.) is updated whether or not the frame is shown.
    fn decode_one_frame(&mut self, frame_data: &[u8]) -> Result<DecodedFrame, DecodeError> {
        let _total_t = bench_timing::StageTimer::start(bench_timing::Stage::Total);
        let (parsed, consumed) = {
            let _t = bench_timing::StageTimer::start(bench_timing::Stage::HeaderParse);
            parse_uncompressed_header(frame_data, &self.persist)?
        };
        let header = match parsed {
            FrameHeader::New(h) => h,
            FrameHeader::ShowExistingFrame {
                frame_to_show_map_idx,
            } => {
                let data = self
                    .dpb
                    .get(frame_to_show_map_idx)
                    .ok_or(DecodeError::MissingReferenceFrame)?;
                return Ok(DecodedFrame {
                    info: None,
                    frame: Some(ref_frame_data_to_frame(data)),
                });
            }
        };

        let color_config = self.resolve_color_config(&header);

        let compressed_start = consumed;
        let compressed_end = compressed_start
            .checked_add(header.header_size_in_bytes as usize)
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

        // Reject an absurd frame size before it reaches the buffer allocations below (which
        // use `vec![]` and would abort the process on failure, not return an error).
        if header.width as u64 * header.height as u64 > header::MAX_FRAME_LUMA_SAMPLES {
            return Err(DecodeError::Header(HeaderError::FrameSizeTooLarge));
        }

        let image_size = header::compute_image_size(header.width, header.height);
        self.clear_prev_segment_ids_if_needed(
            (header.width, header.height),
            &image_size,
            header.frame_is_intra || header.error_resilient_mode,
        );
        self.reset_frame_contexts_if_needed(&header);

        // Load the frame context slot once to seed forward updates in `parse_compressed_header`
        // (by value, avoiding a second ~3.5 KB clone on non-refresh, error-resilient, and
        // frame-parallel frames). When backward adaptation runs after tile decode, the slot in
        // `self.frame_contexts` is still untouched and is loaded then.
        let ctx_idx = header.effective_frame_context_idx();
        let compressed = {
            let _t = bench_timing::StageTimer::start(bench_timing::Stage::CompressedHeader);
            parse_compressed_header(compressed_bytes, &header, self.frame_contexts.load(ctx_idx))?
        };

        let resolved_refs = self.resolve_and_validate_refs(&header, &color_config)?;

        let tile_data = &frame_data[compressed_end..];
        let prev_grid = if use_prev_frame_mvs {
            self.prev_mi_grid.clone()
        } else {
            None
        };
        let y_w = (image_size.sb64_cols * 64) as usize;
        let y_h = (image_size.sb64_rows * 64) as usize;
        let uv_w = y_w >> color_config.subsampling_x;
        let uv_h = y_h >> color_config.subsampling_y;
        let recycled = tile::RecycledBuffers {
            planes: self.take_matching_planes(y_w, y_h, uv_w, uv_h),
            mi_grid: self.mi_grid_pool.take(),
            above_ctx: self.above_ctx_pool.take(),
            scratch: self.tile_scratch_pool.take(),
            counts: self.counts_pool.take(),
        };
        let backward_adapt = header.refresh_frame_context
            && !header.error_resilient_mode
            && !header.frame_parallel_decoding_mode;
        let mut tile_decoder = TileDecoder::new_with_buffers(
            &header,
            color_config,
            &compressed,
            use_prev_frame_mvs,
            prev_grid,
            resolved_refs,
            self.prev_segment_ids.clone(),
            backward_adapt,
            recycled,
        );
        {
            let _t = bench_timing::StageTimer::start(bench_timing::Stage::TileDecode);
            tile_decoder.decode_tiles_with_pool(
                tile_data,
                &mut self.thread_pool,
                &mut self.tile_worker_pool,
            )?;
        }
        // Spec §8.1 step 2: "If loop_filter_level is not equal to 0, the loop filter
        // process ... is invoked" -- gated on the frame-level value, not per-edge deltas.
        if header.loop_filter.level != 0 {
            let _t = bench_timing::StageTimer::start(bench_timing::Stage::LoopFilter);
            tile_decoder.apply_loop_filter_with_pool(&header.loop_filter, &mut self.thread_pool);
        }
        // Remainder of decode_one_frame (PrevSegmentIds refresh, probability adaptation,
        // DPB update, output Frame construction) is timed as one "DpbOutput" bucket; this
        // timer lives to the end of the function (no more fallible `?` calls follow).
        let _dpb_t = bench_timing::StageTimer::start(bench_timing::Stage::DpbOutput);

        self.refresh_prev_segment_ids(&header, &image_size, tile_decoder.mi_grid());

        // Backward adaptation (spec §6.1.2) needs `tile_decoder.counts()` before `tile_decoder`
        // is consumed; FPM / error-resilient refresh saves `compressed.probs` directly after
        // `tile_decoder.into_parts()` drops its `Arc` so `Arc::unwrap_or_clone` avoids cloning.
        if backward_adapt {
            let starting_probs = self.frame_contexts.load(ctx_idx);
            let final_probs = refresh_probs(
                &header,
                starting_probs,
                &compressed,
                tile_decoder.counts(),
                self.last_frame_type,
            );
            self.frame_contexts.save(ctx_idx, final_probs);
        }

        self.update_persistent_state(&header);
        let info = build_frame_decode_info(&header);

        // Last use of tile_decoder: consume it to drop its read-only `Arc`s to DPB slots,
        // `prev_mi_grid`, and `compressed.probs` BEFORE refreshing DPB / saving probs /
        // replacing `self.prev_mi_grid`, so evicted buffers have strong count 1 and recycle
        // cleanly into our pools.
        let (planes, mi_grid, above_ctx, scratch, counts) = tile_decoder.into_parts();
        self.above_ctx_pool = Some(above_ctx);
        self.tile_scratch_pool = Some(scratch);
        if let Some(c) = counts {
            self.counts_pool = Some(c);
        }
        if header.refresh_frame_context && !backward_adapt {
            self.frame_contexts
                .save(ctx_idx, Arc::unwrap_or_clone(compressed.probs));
        }
        let frame = self.output_and_refresh_refs(&header, &color_config, planes);
        self.store_and_recycle_mi_grid(mi_grid);
        Ok(DecodedFrame {
            info: Some(info),
            frame,
        })
    }
}

fn build_frame_decode_info(header: &header::NewFrameHeader) -> FrameDecodeInfo {
    FrameDecodeInfo {
        intra_only: header.intra_only,
        frame_is_intra: header.frame_is_intra,
        reset_frame_context: header.reset_frame_context,
        segmentation_enabled: header.segmentation.enabled,
        seg_features_active: std::array::from_fn(|level| {
            header.segmentation.enabled
                && header
                    .segmentation
                    .feature_enabled
                    .iter()
                    .any(|seg| seg[level])
        }),
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

/// `frame_context_idx` must be the raw bitstream value (`NewFrameHeader::frame_context_idx`, not
/// `effective_frame_context_idx()`), since `save_probs( frame_context_idx )` runs before
/// `setup_past_independence()` forces the effective index to 0.
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

/// `refresh_probs()` (spec §6.1.2): the frame context to save back after a frame's decode --
/// a pure function of the frame's header, the pre-forward-update starting probabilities
/// (`load_probs`' value), the parsed compressed header, the tile decode's adaptation counts,
/// and the previous frame's type (spec §8.4.3's updateFactor). Backward adaptation runs only
/// when neither error_resilient_mode nor frame_parallel_decoding_mode is set; otherwise the
/// post-forward-update probabilities are saved as-is.
fn refresh_probs(
    header: &header::NewFrameHeader,
    starting_probs: FrameContext,
    compressed: &compressed_header::CompressedHeader,
    counts: &counts::Counts,
    last_frame_type: Option<FrameType>,
) -> FrameContext {
    if !header.error_resilient_mode && !header.frame_parallel_decoding_mode {
        // load_probs( frame_context_idx ): restore all tables except tx_probs/skip_prob
        // to their pre-forward-update value (starting_probs). tx_probs/skip_prob are
        // left at their post-forward-update value from compressed_header(). The two
        // fields are copied aside (24B + 3B) before moving starting_probs into working,
        // rather than cloning the whole multi-KB struct just to overwrite most of it.
        let pre_update_tx_probs = starting_probs.tx_probs;
        let pre_update_skip_prob = starting_probs.skip_prob;
        let mut working = starting_probs;
        working.tx_probs = compressed.probs.tx_probs;
        working.skip_prob = compressed.probs.skip_prob;

        // Spec §8.4.3: determining updateFactor.
        let update_factor = if header.frame_is_intra {
            112
        } else if last_frame_type == Some(FrameType::KeyFrame) {
            128
        } else {
            112
        };
        adapt_coef_probs(&mut working.coef_probs, counts, update_factor);

        if !header.frame_is_intra {
            // load_probs2( frame_context_idx ): also restore tx_probs/skip_prob to
            // their pre-forward-update value before applying adapt_noncoef_probs.
            working.tx_probs = pre_update_tx_probs;
            working.skip_prob = pre_update_skip_prob;
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
        (*compressed.probs).clone()
    }
}

#[cfg(test)]
#[path = "../tests/unit/lib.rs"]
mod tests;
