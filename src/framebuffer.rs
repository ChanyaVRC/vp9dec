//! Plane representation of the frame buffer (`CurrFrame`).
//!
//! The spec treats the frame buffer as a 3-dimensional array `CurrFrame[ plane ][ y ][ x ]`,
//! but this implementation holds each plane as a 1-dimensional `Vec<u16>` (row-major, with an
//! explicit `stride >= width` so cropped DPB reference planes can reuse superblock-padded
//! backing buffers in-place without copying).
//!
//! `u16` regardless of `BitDepth`: 8-bit samples (0..=255) are stored widened rather than
//! packed, so every plane operation (get/set/loop filter/prediction) has exactly one code
//! path across all 3 bit depths -- narrowing to `u8` happens only at the output boundary
//! (see `crop_u8`, used for the `Frame`/`PlaneData::U8` output when `BitDepth == 8`).
//!
//! The decode working buffer is allocated not at the display size (`FrameWidth`/`FrameHeight`)
//! but at the size rounded up to the superblock boundary (`Sb64Cols*64`/`Sb64Rows*64`, chroma
//! planes after subsampling). This is because the block boundary handling of spec §6.4.21
//! `residual()` can cause blocks at the frame edge to write slightly beyond `(MiCols*8, MiRows*8)`.

/// A single plane's buffer. Samples are stored as `u16` for every `BitDepth` (8/10/12-bit);
/// see the module doc for why.
///
/// A plane normally covers the whole frame (`x0 == 0`). A tile-parallel worker instead holds a
/// *column strip* ([`Plane::new_strip`]): a buffer covering only the absolute pixel columns
/// `[x0, x0 + width)`. All accessors keep taking **absolute** frame x coordinates -- the strip
/// origin is subtracted internally -- so the decode path is identical either way.
#[derive(Debug, Clone)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    /// Row stride in samples (`>= width`). Equals `width` for freshly allocated planes; may
    /// exceed `width` after [`Plane::crop_in_place`] so DPB reference storage avoids copying.
    stride: usize,
    /// Absolute pixel column of this buffer's first column (0 for a whole-frame plane).
    pub x0: usize,
    data: Vec<u16>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self::new_strip(width, height, 0)
    }

    /// A column strip covering absolute pixel columns `[x0, x0 + width)` of a conceptual
    /// larger plane (used by the tile-parallel worker decoders, `tile::spawn_column_worker`).
    /// Accessors take absolute x; `x` must stay within the strip. The whole-frame `crop*`
    /// outputs are not meaningful on a strip.
    pub fn new_strip(width: usize, height: usize, x0: usize) -> Self {
        Self {
            width,
            height,
            stride: width,
            x0,
            data: vec![0u16; width * height],
        }
    }

    /// Row stride in samples (`>= width`). Use this (not `width`) when indexing [`Self::as_slice`]
    /// or [`Self::as_mut_slice`] directly.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Backing allocation geometry `(stride, len)` for pool lookup and deduplication.
    #[inline]
    pub(crate) fn backing_dims(&self) -> (usize, usize) {
        (self.stride, self.data.len())
    }

    /// Crops this plane's logical `(width, height)` in-place without reallocating or copying
    /// the backing buffer (for zero-copy DPB storage in `RefFrameData`).
    pub(crate) fn crop_in_place(&mut self, crop_width: usize, crop_height: usize) {
        debug_assert_eq!(self.x0, 0, "crop_in_place requires a whole-frame plane");
        debug_assert!(crop_width <= self.stride);
        debug_assert!(crop_height * self.stride <= self.data.len());
        self.width = crop_width;
        self.height = crop_height;
    }

    /// Restores a pooled plane back to full `(width, height)` with `x0 == 0` if its backing
    /// allocation matches `(stride == width && data.len() == width * height)`. Returns `true`
    /// on match so the caller can reuse it without allocating.
    pub(crate) fn try_reuse_as(&mut self, width: usize, height: usize) -> bool {
        self.try_reuse_as_strip(width, height, 0)
    }

    /// Reuses a pooled whole-frame working plane for `(width, height)` when its backing capacity
    /// fits within `[width * height, max_keep_elems]`, avoiding reallocations across multi-layer
    /// SVC while releasing oversized buffers after a permanent down-resize.
    pub(crate) fn try_reuse_working(
        &mut self,
        width: usize,
        height: usize,
        max_keep_elems: usize,
    ) -> bool {
        let need = width * height;
        if self.data.capacity() >= need && self.data.capacity() <= max_keep_elems.max(need) {
            self.width = width;
            self.height = height;
            self.stride = width;
            self.x0 = 0;
            self.data.resize(need, 0);
            true
        } else {
            false
        }
    }

    /// Reuses a pooled column-strip plane for `(width, height, x0)` if its backing allocation
    /// matches `(stride == width && data.len() == width * height)` (zero allocation/zeroing).
    pub(crate) fn try_reuse_as_strip(&mut self, width: usize, height: usize, x0: usize) -> bool {
        if self.stride == width && self.data.len() == width * height {
            self.width = width;
            self.height = height;
            self.x0 = x0;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u16 {
        debug_assert!(
            x >= self.x0 && x - self.x0 < self.width && y < self.height,
            "Plane::get out of range"
        );
        self.data[y * self.stride + (x - self.x0)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u16) {
        debug_assert!(
            x >= self.x0 && x - self.x0 < self.width && y < self.height,
            "Plane::set out of range"
        );
        self.data[y * self.stride + (x - self.x0)] = v;
    }

    /// Raw row-major buffer (row stride == [`Self::stride`]), for the AVX2 SIMD kernels.
    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }

    /// Mutable counterpart of [`Plane::as_slice`].
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u16] {
        &mut self.data
    }

    /// Returns a row-major sample sequence cropped to the display size `(crop_width, crop_height)`.
    pub fn crop(&self, crop_width: usize, crop_height: usize) -> Vec<u16> {
        debug_assert_eq!(self.x0, 0, "crop is only meaningful on a whole-frame plane");
        if crop_width == self.stride {
            return self.data[..crop_width * crop_height].to_vec();
        }
        let mut out = Vec::with_capacity(crop_width * crop_height);
        for y in 0..crop_height {
            let row_start = y * self.stride;
            out.extend_from_slice(&self.data[row_start..row_start + crop_width]);
        }
        out
    }

    /// Same as [`Plane::crop`], narrowed to `u8` (for `PlaneData::U8` output when
    /// `BitDepth == 8`, where every sample is already known to fit in `0..=255`).
    pub fn crop_u8(&self, crop_width: usize, crop_height: usize) -> Vec<u8> {
        let mut out = Vec::new();
        self.narrow_u8_into(crop_width, crop_height, &mut out);
        out
    }

    /// Narrows the cropped `(crop_width, crop_height)` region of this plane to `u8` into `out`,
    /// reusing `out`'s allocation when its capacity already suffices.
    pub(crate) fn narrow_u8_into(&self, crop_width: usize, crop_height: usize, out: &mut Vec<u8>) {
        debug_assert_eq!(self.x0, 0, "crop is only meaningful on a whole-frame plane");
        debug_assert!(crop_width <= self.stride);
        debug_assert!(crop_height * self.stride <= self.data.len());
        let total = crop_width * crop_height;
        out.clear();
        out.reserve(total);

        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2_enabled() {
            let dst_base = out.spare_capacity_mut().as_mut_ptr() as *mut u8;
            if crop_width == self.stride {
                // Single contiguous narrow pass when display width matches plane stride
                // (e.g. 1920x1080, 1920x800, 1280x720).
                // SAFETY: `avx2_enabled()` confirmed; `out` has capacity >= `total` == `src.len()`.
                unsafe {
                    crate::simd::narrow_u16_to_u8_avx2(&self.data[..total], dst_base);
                    out.set_len(total);
                }
                return;
            }
            for y in 0..crop_height {
                let row_start = y * self.stride;
                let src_row = &self.data[row_start..row_start + crop_width];
                // SAFETY: `avx2_enabled()` confirmed; row slice fits `out`'s reserved capacity.
                unsafe {
                    crate::simd::narrow_u16_to_u8_avx2(src_row, dst_base.add(y * crop_width));
                }
            }
            // SAFETY: all `crop_height * crop_width == total` bytes were written above.
            unsafe {
                out.set_len(total);
            }
            return;
        }

        for y in 0..crop_height {
            let row_start = y * self.stride;
            out.extend(
                self.data[row_start..row_start + crop_width]
                    .iter()
                    .map(|&v| v as u8),
            );
        }
    }

    /// Same as [`Plane::crop`] but returns a new [`Plane`] instead of a `Vec<u16>`
    /// (for storing into `FrameStore` / DPB reference frame data per spec §8.10).
    pub fn crop_to_plane(&self, crop_width: usize, crop_height: usize) -> Plane {
        Plane {
            width: crop_width,
            height: crop_height,
            stride: crop_width,
            x0: 0,
            data: self.crop(crop_width, crop_height),
        }
    }
}

/// Backing pixel storage for a cropped DPB reference plane ([`RefPlane`]).
///
/// 8-bit streams (profiles 0/1) store packed `u8` samples (halving DPB footprint and doubling
/// cache density during motion compensation); 10/12-bit streams (profiles 2/3) keep `u16`.
#[derive(Debug, Clone)]
pub enum RefPlaneData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// One cropped reference plane stored inside [`crate::dpb::RefFrameData`].
#[derive(Debug, Clone)]
pub struct RefPlane {
    pub width: usize,
    pub height: usize,
    stride: usize,
    data: RefPlaneData,
}

impl RefPlane {
    /// Wraps a tightly packed `width x height` 8-bit reference plane (`stride == width`).
    pub(crate) fn from_u8_cropped(width: usize, height: usize, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), width * height);
        Self {
            width,
            height,
            stride: width,
            data: RefPlaneData::U8(data),
        }
    }

    /// Moves a cropped-in-place 10/12-bit [`Plane`] into a `RefPlane` with zero copy.
    pub(crate) fn from_u16_plane(plane: Plane) -> Self {
        debug_assert_eq!(plane.x0, 0);
        Self {
            width: plane.width,
            height: plane.height,
            stride: plane.stride,
            data: RefPlaneData::U16(plane.data),
        }
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub fn data(&self) -> &RefPlaneData {
        &self.data
    }

    /// Backing geometry `(stride, len)` when `U16` (for `plane_pool` matching).
    #[inline]
    pub(crate) fn u16_backing_dims(&self) -> Option<(usize, usize)> {
        match &self.data {
            RefPlaneData::U16(v) => Some((self.stride, v.len())),
            RefPlaneData::U8(_) => None,
        }
    }

    /// Cropped sample count `width * height` when `U8` (for `ref_u8_pool` matching).
    #[inline]
    pub(crate) fn u8_len(&self) -> Option<usize> {
        match &self.data {
            RefPlaneData::U8(v) => Some(v.len()),
            RefPlaneData::U16(_) => None,
        }
    }

    /// Unwraps the owned backing buffer for caller-managed DPB eviction pooling (`Ok(Vec<u8>)`
    /// for 8-bit, `Err(Plane)` for 10/12-bit).
    pub(crate) fn into_backing(self) -> Result<Vec<u8>, Plane> {
        match self.data {
            RefPlaneData::U8(v) => Ok(v),
            RefPlaneData::U16(data) => Err(Plane {
                width: self.width,
                height: self.height,
                stride: self.stride,
                x0: 0,
                data,
            }),
        }
    }

    /// Copies out display-cropped [`crate::PlaneData`] (used by `show_existing_frame`).
    pub(crate) fn to_plane_data(&self) -> crate::PlaneData {
        match &self.data {
            RefPlaneData::U8(v) => {
                debug_assert_eq!(self.stride, self.width);
                crate::PlaneData::U8(v.clone())
            }
            RefPlaneData::U16(v) => {
                if self.width == self.stride {
                    crate::PlaneData::U16(v[..self.width * self.height].to_vec())
                } else {
                    let mut out = Vec::with_capacity(self.width * self.height);
                    for y in 0..self.height {
                        let start = y * self.stride;
                        out.extend_from_slice(&v[start..start + self.width]);
                    }
                    crate::PlaneData::U16(out)
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/framebuffer.rs"]
mod tests;
