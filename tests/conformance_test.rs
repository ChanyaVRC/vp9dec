//! 公式 VP9 コンフォーマンステストベクタによる、ビット完全一致の検証（M2b）。
//!
//! libvpx のテストデータ配布 (`https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`)
//! には各 `.ivf` に対応する `.ivf.md5` が同梱されており、各出力フレーム（1 行 = 1 フレーム）の
//! I420（Y 全体 → U 全体 → V 全体を連結したもの）の MD5 が記録されている
//! （`md5sum` 標準フォーマット: `<32文字hex>  <ファイル名>`）。
//!
//! ここでは最初の出力フレーム（= 最初のキーフレーム）について、`decode_keyframe` の出力
//! （表示サイズにクロップ済みの Y→U→V 連結バイト列）の MD5 が `.ivf.md5` の 1 行目と
//! 完全一致することを検証する。ループフィルタ・クロップ・プレーン連結順・レンダーサイズの
//! 解釈のいずれかが誤っていれば、この比較は確実に失敗する。
//!
//! テストベクタ・MD5 ファイルは `tests/vectors/` にダウンロード済みであることを前提とする
//! （`.gitignore` 対象。取得手順は README.md 参照）。存在しない場合は早期 return + `eprintln!`
//! でスキップする。

use std::path::Path;

use vp9dec::ivf::IvfReader;
use vp9dec::md5::{md5, to_hex};
use vp9dec::{decode_keyframe, Decoder, Frame};

/// `.ivf.md5` の 1 行目から MD5 16進文字列だけを取り出す。
/// フォーマットは `md5sum` 互換: `<hex>␠␠<filename>`（区切りは 2 スペースまたはタブ）。
fn first_line_md5(md5_file_contents: &str) -> &str {
    let first_line = md5_file_contents
        .lines()
        .next()
        .expect(".ivf.md5 ファイルが空");
    first_line
        .split_whitespace()
        .next()
        .expect(".ivf.md5 の1行目に空白区切りのフィールドがない")
}

/// `Frame` を I420 として Y→U→V の順に連結したバイト列（`.ivf.md5` が期待する並び）を返す。
fn i420_bytes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.y.len() + frame.u.len() + frame.v.len());
    out.extend_from_slice(&frame.y);
    out.extend_from_slice(&frame.u);
    out.extend_from_slice(&frame.v);
    out
}

fn check_vector(ivf_name: &str) {
    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors");
    let ivf_path = vectors_dir.join(ivf_name);
    let md5_path = vectors_dir.join(format!("{ivf_name}.md5"));

    if !ivf_path.exists() || !md5_path.exists() {
        eprintln!(
            "[skip] テストベクタまたは .ivf.md5 が見つからないためスキップします: {} / {}\n\
             README.md の手順に従って事前にダウンロードしてください。",
            ivf_path.display(),
            md5_path.display()
        );
        return;
    }

    let ivf_bytes = std::fs::read(&ivf_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ivf_path.display()));
    let md5_text = std::fs::read_to_string(&md5_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", md5_path.display()));
    let expected = first_line_md5(&md5_text).to_ascii_lowercase();

    let mut reader = IvfReader::new(&ivf_bytes).unwrap_or_else(|e| {
        panic!(
            "failed to parse IVF header for {}: {e:?}",
            ivf_path.display()
        )
    });
    let first_frame = reader
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", ivf_path.display()))
        .unwrap_or_else(|e| {
            panic!(
                "failed to read first frame of {}: {e:?}",
                ivf_path.display()
            )
        });

    let frame = decode_keyframe(first_frame.data).unwrap_or_else(|e| {
        panic!(
            "{}: decode_keyframe failed on first frame: {e:?}",
            ivf_path.display()
        )
    });

    let actual_bytes = i420_bytes(&frame);
    let actual = to_hex(&md5(&actual_bytes));

    assert_eq!(
        actual,
        expected,
        "{}: 第1フレーム(キーフレーム)のMD5が公式値と不一致\n  actual:   {}\n  expected: {}\n\
         (frame: {}x{}, y.len={}, u.len={}, v.len={})",
        ivf_path.display(),
        actual,
        expected,
        frame.width,
        frame.height,
        frame.y.len(),
        frame.u.len(),
        frame.v.len(),
    );
    eprintln!(
        "[ok] {}: 第1フレームのMD5が公式値と完全一致 ({})",
        ivf_path.display(),
        actual
    );
}

#[test]
fn vp90_2_12_droppable_1_first_keyframe_matches_official_md5() {
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_first_keyframe_matches_official_md5() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}

/// M3 後半: **全表示フレーム**について `.ivf.md5` の該当行と完全一致することを検証する
/// （動き補償・確率適応・DPB・ループフィルタのフレーム間状態がすべて正しくないと通らない、
/// キーフレーム単体の検証よりはるかに厳しいテスト）。
///
/// `.ivf.md5` は「1 行 = 1 出力（表示）フレーム」であり、[`Decoder::decode_frame`] が
/// `Some(Frame)` を返すたびに 1 行ずつ消費して比較する（`show_frame == 0` の非表示フレームは
/// `.ivf.md5` に対応する行を持たないため、`None` が返っても行を消費しない）。
fn check_all_frames(ivf_name: &str) {
    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors");
    let ivf_path = vectors_dir.join(ivf_name);
    let md5_path = vectors_dir.join(format!("{ivf_name}.md5"));

    if !ivf_path.exists() || !md5_path.exists() {
        eprintln!(
            "[skip] テストベクタまたは .ivf.md5 が見つからないためスキップします: {} / {}\n\
             README.md の手順に従って事前にダウンロードしてください。",
            ivf_path.display(),
            md5_path.display()
        );
        return;
    }

    let ivf_bytes = std::fs::read(&ivf_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ivf_path.display()));
    let md5_text = std::fs::read_to_string(&md5_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", md5_path.display()));
    let expected_lines: Vec<String> = md5_text
        .lines()
        .map(|line| {
            line.split_whitespace()
                .next()
                .unwrap_or_else(|| panic!("{}: 空行がある", md5_path.display()))
                .to_ascii_lowercase()
        })
        .collect();

    let reader = IvfReader::new(&ivf_bytes).unwrap_or_else(|e| {
        panic!(
            "failed to parse IVF header for {}: {e:?}",
            ivf_path.display()
        )
    });

    let mut decoder = Decoder::new();
    let mut output_idx = 0usize;
    let mut mismatches: Vec<usize> = Vec::new();

    for (ivf_frame_idx, frame) in reader.enumerate() {
        let frame = frame.unwrap_or_else(|e| {
            panic!(
                "{}: failed to read IVF frame {ivf_frame_idx}: {e:?}",
                ivf_path.display()
            )
        });
        let outcome = decoder.decode_frame(frame.data).unwrap_or_else(|e| {
            panic!(
                "{}: IVF frame {ivf_frame_idx} failed to decode: {e:?}",
                ivf_path.display()
            )
        });
        if let Some(decoded) = outcome {
            let actual_bytes = i420_bytes(&decoded);
            let actual = to_hex(&md5(&actual_bytes));
            let expected = expected_lines.get(output_idx).unwrap_or_else(|| {
                panic!(
                    "{}: .ivf.md5 の行数({})を超える出力フレームが生成された（output_idx={output_idx}）",
                    ivf_path.display(),
                    expected_lines.len()
                )
            });
            if &actual != expected {
                mismatches.push(output_idx);
                eprintln!(
                    "[NG] {}: output frame {output_idx} (ivf frame {ivf_frame_idx}) MD5 不一致\n  actual:   {actual}\n  expected: {expected}\n  ({}x{}, y.len={}, u.len={}, v.len={})",
                    ivf_path.display(),
                    decoded.width,
                    decoded.height,
                    decoded.y.len(),
                    decoded.u.len(),
                    decoded.v.len(),
                );
                // 最初の不一致だけ詳しく調べれば十分なので、以降は早期終了する
                // （README「デバッグ指針」参照: まず何フレーム目まで合うかを見る）。
                break;
            }
            output_idx += 1;
        }
    }

    assert!(
        mismatches.is_empty(),
        "{}: {output_idx} フレーム目までは一致、{}番目の出力フレームで不一致（全 {} 出力フレーム中）",
        ivf_path.display(),
        mismatches[0],
        expected_lines.len()
    );
    assert_eq!(
        output_idx,
        expected_lines.len(),
        "{}: 出力フレーム数が .ivf.md5 の行数と一致しない(全フレーム比較できていない可能性)",
        ivf_path.display()
    );
    eprintln!(
        "[ok] {}: 全 {output_idx} 出力フレームが公式 MD5 と完全一致",
        ivf_path.display()
    );
}

#[test]
fn vp90_2_12_droppable_1_all_frames_match_official_md5() {
    check_all_frames("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_all_frames_match_official_md5() {
    check_all_frames("vp90-2-09-subpixel-00.ivf");
}
