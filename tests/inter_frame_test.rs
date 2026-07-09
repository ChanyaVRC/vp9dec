//! インターフレームのビットストリーム復号（M3 前半）の統合テスト。
//!
//! `tests/vectors/` にダウンロード済みの `.ivf` ファイルがあれば、ファイル内のすべての
//! フレーム（キーフレーム・インターフレーム・droppable フレームを含む）を [`Decoder`] で
//! 順番にデコードし、`uncompressed_header` + `compressed_header` + 全タイルのモード情報/MV/
//! 残差トークンを最後までパニックなく読み切れることを検証する。
//!
//! 画素生成（動き補償・サブピクセル補間）は M3 後半で実装予定のためまだスタブであり、ここでは
//! ピクセル値の正しさは検証しない（`decode_test.rs`/`conformance_test.rs` がキーフレームの
//! ピクセル正しさを別途検証している）。ビットストリームを最後まで正しく読み切れていることは、
//! 後続フレームも連続して（bool デコーダの消費位置がずれずに）読めることで担保される
//! （1 フレームでも消費位置がずれれば、次のフレームの `uncompressed_header` の
//! `frame_marker`/`frame_sync_code` 等の検証で早期に失敗するはず）。
//!
//! テストベクタが存在しない環境では、該当テストは早期 return + `eprintln!` でスキップされる
//! （取得方法は README.md を参照）。

use std::path::Path;

use vp9dec::ivf::IvfReader;
use vp9dec::Decoder;

fn check_vector(relative_path: &str) {
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
    let reader = IvfReader::new(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse IVF header for {}: {e:?}", path.display()));

    let mut decoder = Decoder::new();
    let mut frame_count = 0usize;
    let mut decoded_count = 0usize;
    let mut hidden_count = 0usize;

    for (i, frame) in reader.enumerate() {
        let frame = frame
            .unwrap_or_else(|e| panic!("{}: failed to read IVF frame {i}: {e:?}", path.display()));
        match decoder.decode_frame(frame.data) {
            Ok(Some(f)) => {
                decoded_count += 1;
                // 最低限の健全性: プレーンサイズが frame.width/height から導出した想定値と一致する。
                assert_eq!(
                    f.y.len(),
                    (f.width * f.height) as usize,
                    "{}: frame {i}: unexpected Y plane size",
                    path.display()
                );
            }
            Ok(None) => {
                hidden_count += 1;
            }
            Err(e) => panic!(
                "{}: frame {i} (of {} total so far) failed to decode: {e:?}",
                path.display(),
                frame_count + 1
            ),
        }
        frame_count += 1;
    }

    eprintln!(
        "[ok] {}: {frame_count} IVF フレームすべてを読み切った（表示フレーム {decoded_count} 件、非表示フレーム {hidden_count} 件）",
        path.display()
    );
    assert!(
        frame_count > 1,
        "{}: 単一フレームしかない（インターフレームの検証にならない）",
        path.display()
    );
    assert!(
        decoded_count > 1,
        "{}: 新規デコードされたフレームが 1 枚以下（インターフレームが含まれていない可能性）",
        path.display()
    );
}

#[test]
fn vp90_2_12_droppable_1_reads_all_frames_to_completion() {
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_reads_all_frames_to_completion() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}
