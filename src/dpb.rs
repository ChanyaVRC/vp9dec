//! 参照フレームバッファ（DPB, Decoded Picture Buffer）管理（仕様 8.10 節
//! "Reference frame update process" ＋ 8.9 節 "Output process" の `show_existing_frame` 分岐）。
//!
//! 仕様の `FrameStore[ i ][ plane ]`/`RefFrameWidth[ i ]`/`RefFrameHeight[ i ]`/
//! `RefSubsamplingX[ i ]`/`RefSubsamplingY[ i ]`/`RefBitDepth[ i ]` を 8 スロットぶん保持する。
//! `FrameStore` は表示サイズ（`FrameWidth`/`FrameHeight`、クロマはサブサンプリング後）に
//! クロップ済みのピクセルデータを保持する（仕様 8.10 節のコピー範囲が `x = 0..FrameWidth-1`
//! 等になっていることに対応）。これにより、動き補償のクランプ処理（仕様 8.5.2.4 節の
//! `lastX`/`lastY`）がプレーンの `width - 1`/`height - 1` とそのまま一致する。

use crate::framebuffer::Plane;
use crate::prob_tables::NUM_REF_FRAMES;

/// 1 スロットぶんの参照フレームデータ（仕様の `FrameStore[ i ]` + 付随するサイズ情報）。
#[derive(Debug, Clone)]
pub struct RefFrameData {
    /// `RefFrameWidth[ i ]`（輝度基準のフレーム幅）。
    pub width: u32,
    /// `RefFrameHeight[ i ]`。
    pub height: u32,
    pub subsampling_x: u32,
    pub subsampling_y: u32,
    pub bit_depth: u8,
    /// クロップ済み（`width x height`）の輝度プレーン。
    pub y: Plane,
    /// クロップ済み（`((width+subX)>>subX) x ((height+subY)>>subY)`）の色差プレーン。
    pub u: Plane,
    pub v: Plane,
}

/// 8 スロットの DPB。
#[derive(Debug, Clone)]
pub struct Dpb {
    slots: [Option<RefFrameData>; NUM_REF_FRAMES],
}

impl Default for Dpb {
    fn default() -> Self {
        Self::new()
    }
}

impl Dpb {
    pub fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }

    pub fn get(&self, idx: u8) -> Option<&RefFrameData> {
        self.slots[idx as usize].as_ref()
    }

    /// `Reference frame update process`（仕様 8.10 節）のステップ 1。
    /// `refresh_frame_flags` のビットが立っているスロットすべてに `data` の複製を書き込む。
    pub fn update(&mut self, refresh_frame_flags: u8, data: &RefFrameData) {
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            if (refresh_frame_flags >> slot) & 1 == 1 {
                *entry = Some(data.clone());
            }
        }
    }
}
