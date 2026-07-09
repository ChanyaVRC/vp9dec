//! `decode_keyframe`（M2 の公開 API）の統合テスト。
//!
//! `tests/vectors/` にダウンロード済みの実データ（キーフレームを含む VP9 ストリーム）を使い、
//! 最初のキーフレームが最後までデコードでき、出力された Y プレーンが単調でない
//! （実写系ベクタとして妥当な統計値を持つ）ことを検証する。
//!
//! テストベクタが存在しない環境では、該当テストは早期 return + `eprintln!` でスキップされる
//! （取得方法は README.md を参照）。

use std::path::Path;

use vp9dec::ivf::IvfReader;
use vp9dec::{decode_keyframe, Frame};

struct YStats {
    min: u8,
    max: u8,
    mean: f64,
    variance: f64,
    all_same: bool,
}

fn y_stats(frame: &Frame) -> YStats {
    let min = *frame.y.iter().min().expect("non-empty Y plane");
    let max = *frame.y.iter().max().expect("non-empty Y plane");
    let n = frame.y.len() as f64;
    let mean = frame.y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = frame
        .y
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    YStats {
        min,
        max,
        mean,
        variance,
        all_same: min == max,
    }
}

fn check_vector(relative_path: &str, expected_width: u32, expected_height: u32) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join(relative_path);

    if !path.exists() {
        eprintln!(
            "[skip] テストベクタが見つからないためスキップします: {}\n\
             README.md の手順に従って事前にダウンロードしてください。",
            path.display()
        );
        return;
    }

    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read test vector {}: {e}", path.display()));
    let mut reader = IvfReader::new(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse IVF header for {}: {e:?}", path.display()));
    let first_frame = reader
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let frame = decode_keyframe(first_frame.data).unwrap_or_else(|e| {
        panic!(
            "{}: decode_keyframe failed on first frame: {e:?}",
            path.display()
        )
    });

    assert_eq!(
        frame.width,
        expected_width,
        "{}: unexpected width",
        path.display()
    );
    assert_eq!(
        frame.height,
        expected_height,
        "{}: unexpected height",
        path.display()
    );
    assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);

    let uv_w = (frame.width as usize).div_ceil(2);
    let uv_h = (frame.height as usize).div_ceil(2);
    assert_eq!(frame.u.len(), uv_w * uv_h);
    assert_eq!(frame.v.len(), uv_w * uv_h);

    let stats = y_stats(&frame);
    eprintln!(
        "[ok] {}: {}x{}, Y min={} max={} mean={:.2} variance={:.2}",
        path.display(),
        frame.width,
        frame.height,
        stats.min,
        stats.max,
        stats.mean,
        stats.variance
    );

    assert!(
        !stats.all_same,
        "{}: Y プレーンが全ピクセル同値（デコード結果が不自然）",
        path.display()
    );
    assert!(
        stats.variance > 0.0,
        "{}: Y プレーンの分散が 0",
        path.display()
    );
    assert!(
        stats.min < 50 && stats.max > 200,
        "{}: 実写系ベクタとして不自然な輝度レンジ (min={}, max={})",
        path.display(),
        stats.min,
        stats.max
    );
}

#[test]
fn vp90_2_12_droppable_1_decodes_first_keyframe() {
    check_vector("vp90-2-12-droppable_1.ivf", 352, 288);
}

#[test]
fn vp90_2_09_subpixel_00_decodes_first_keyframe() {
    check_vector("vp90-2-09-subpixel-00.ivf", 320, 180);
}
