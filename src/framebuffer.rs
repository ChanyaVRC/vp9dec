//! フレームバッファ（`CurrFrame`）のプレーン表現。
//!
//! 仕様は `CurrFrame[ plane ][ y ][ x ]` という 3 次元配列としてフレームバッファを扱うが、
//! 本実装ではプレーンごとに 1 次元の `Vec<u8>`（行優先）として保持する。
//!
//! バッファのサイズは表示サイズ（`FrameWidth`/`FrameHeight`）ではなく、スーパーブロック
//! 境界（`Sb64Cols*64`/`Sb64Rows*64`、色差プレーンはサブサンプリング後）に切り上げたサイズで
//! 確保する。これは仕様 6.4.21 節 `residual()` のブロック境界処理により、フレーム端の
//! ブロックが `(MiCols*8, MiRows*8)`（= 表示可能な最終クロップ前のサイズ）をわずかに
//! 超えて書き込まれる場合があるため（読み出しは `Min(maxX, ...)` で必ずクリップされるが、
//! 書き込み側の `pred[i][j]`/`Dequant[i][j]` の代入はクリップされない）。

/// 1 プレーン分のバッファ（8bit 固定）。
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

    /// `(x, y)` を `[0, width-1] x [0, height-1]` にクランプしてから読む
    /// （仕様の `CurrFrame[ plane ][ Min(maxY,...) ][ Min(maxX,...) ]` のような
    /// 「フレーム端を超えたら端の値を複製する」参照に使う）。
    #[inline]
    pub fn get_clamped(&self, x: i64, y: i64) -> u8 {
        let cx = x.clamp(0, self.width as i64 - 1) as usize;
        let cy = y.clamp(0, self.height as i64 - 1) as usize;
        self.get(cx, cy)
    }

    /// 表示サイズ `(crop_width, crop_height)` にクロップした行優先バイト列を返す。
    pub fn crop(&self, crop_width: usize, crop_height: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(crop_width * crop_height);
        for y in 0..crop_height {
            let row_start = y * self.width;
            out.extend_from_slice(&self.data[row_start..row_start + crop_width]);
        }
        out
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
