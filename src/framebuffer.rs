//! Plane representation of the frame buffer (`CurrFrame`).
//!
//! The spec treats the frame buffer as a 3-dimensional array `CurrFrame[ plane ][ y ][ x ]`,
//! but this implementation holds each plane as a 1-dimensional `Vec<u8>` (row-major).
//!
//! The buffer is allocated not at the display size (`FrameWidth`/`FrameHeight`) but at
//! the size rounded up to the superblock boundary (`Sb64Cols*64`/`Sb64Rows*64`, chroma
//! planes after subsampling). This is because the block boundary handling of spec
//! §6.4.21 `residual()` can cause blocks at the frame edge to write slightly beyond
//! `(MiCols*8, MiRows*8)` (the size just before the final display crop) (reads are
//! always clipped via `Min(maxX, ...)`, but writes to `pred[i][j]`/`Dequant[i][j]` are not clipped).

/// A single plane's buffer (fixed at 8-bit).
#[derive(Debug, Clone)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    data: Vec<u8>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; width * height],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        debug_assert!(x < self.width && y < self.height, "Plane::get out of range");
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        debug_assert!(x < self.width && y < self.height, "Plane::set out of range");
        self.data[y * self.width + x] = v;
    }

    /// Reads after clamping `(x, y)` to `[0, width-1] x [0, height-1]`
    /// (used for references like the spec's `CurrFrame[ plane ][ Min(maxY,...) ][ Min(maxX,...) ]`
    /// that replicate the edge value past the frame boundary).
    #[inline]
    pub fn get_clamped(&self, x: i64, y: i64) -> u8 {
        let cx = x.clamp(0, self.width as i64 - 1) as usize;
        let cy = y.clamp(0, self.height as i64 - 1) as usize;
        self.get(cx, cy)
    }

    /// Returns a row-major byte sequence cropped to the display size `(crop_width, crop_height)`.
    pub fn crop(&self, crop_width: usize, crop_height: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(crop_width * crop_height);
        for y in 0..crop_height {
            let row_start = y * self.width;
            out.extend_from_slice(&self.data[row_start..row_start + crop_width]);
        }
        out
    }

    /// Same as [`Plane::crop`] but returns a new [`Plane`] instead of a `Vec<u8>`
    /// (for storing into `FrameStore` / DPB reference frame data per spec §8.10).
    pub fn crop_to_plane(&self, crop_width: usize, crop_height: usize) -> Plane {
        Plane {
            width: crop_width,
            height: crop_height,
            data: self.crop(crop_width, crop_height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_set_roundtrip() {
        let mut p = Plane::new(4, 3);
        p.set(1, 2, 42);
        assert_eq!(p.get(1, 2), 42);
        assert_eq!(p.get(0, 0), 0);
    }

    #[test]
    fn get_clamped_clips_to_bounds() {
        let mut p = Plane::new(4, 3);
        p.set(0, 0, 7);
        p.set(3, 2, 9);
        assert_eq!(p.get_clamped(-5, -5), 7);
        assert_eq!(p.get_clamped(100, 100), 9);
        assert_eq!(p.get_clamped(0, 0), 7);
    }

    #[test]
    fn crop_extracts_top_left_region() {
        let mut p = Plane::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                p.set(x, y, (y * 4 + x) as u8);
            }
        }
        let cropped = p.crop(2, 2);
        assert_eq!(cropped, vec![0, 1, 4, 5]);
    }
}
