//! WebM 公式テストベクタを使った、圧縮ヘッダ（`compressed_header`）パースの統合テスト。
//!
//! `tests/vectors/` にダウンロード済みの `.ivf` ファイルがあれば、最初のキーフレームについて
//! 「uncompressed_header → compressed_header がパニックせず最後まで読める」ことと、
//! 読み取った tx_mode / skip_prob が妥当な値域に収まることを検証する。
//!
//! `decode_tiles` はトークン復号・再構成まで含めて完全に実装済みだが、本テストでは
//! 引き続き `compressed_header` の読了までを主目的として検証し、`TileDecoder::decode_tiles`
//! の呼び出しは「パニックしないこと」のみを確認する（成功・失敗どちらの `Result` も許容する）。
//! 完全なピクセル出力の正しさ（統計的な sanity チェック）は `tests/decode_test.rs` の
//! `decode_keyframe` 経由のテストで検証している。タイル/パーティション/モード情報の詳細な
//! 正しさは `src/tile.rs` 内の合成ビットストリームによる単体テストで検証している。
//!
//! テストベクタが存在しない環境では、該当テストは早期 return + `eprintln!` でスキップされる
//! （取得方法は README.md を参照）。

use std::path::Path;

use vp9dec::compressed_header::parse_compressed_header;
use vp9dec::header::{parse_uncompressed_header, FrameHeader};
use vp9dec::ivf::IvfReader;
use vp9dec::tile::TileDecoder;

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

    let mut frames = reader;
    let first_frame = frames
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let (parsed, consumed) = parse_uncompressed_header(first_frame.data).unwrap_or_else(|e| {
        panic!(
            "failed to parse uncompressed header of first frame in {}: {e:?}",
            path.display()
        )
    });

    let header = match parsed {
        FrameHeader::New(h) => h,
        FrameHeader::ShowExistingFrame { .. } => {
            panic!(
                "first frame of {} unexpectedly used show_existing_frame",
                path.display()
            )
        }
    };

    let header_size = header.header_size_in_bytes as usize;
    assert!(
        header_size > 0,
        "{}: header_size_in_bytes should be non-zero for a real key frame",
        path.display()
    );
    let compressed_start = consumed;
    let compressed_end = compressed_start + header_size;
    assert!(
        compressed_end <= first_frame.data.len(),
        "{}: compressed header ({compressed_start}..{compressed_end}) exceeds frame data length {}",
        path.display(),
        first_frame.data.len()
    );
    let compressed_bytes = &first_frame.data[compressed_start..compressed_end];

    let compressed = parse_compressed_header(compressed_bytes, header.quantization.lossless)
        .unwrap_or_else(|e| {
            panic!(
                "{}: failed to parse compressed_header (size={header_size}): {e:?}",
                path.display()
            )
        });

    // tx_mode は ONLY_4X4(0) 〜 TX_MODE_SELECT(4) の範囲に収まる。
    assert!(
        compressed.tx_mode <= 4,
        "{}: tx_mode out of range: {}",
        path.display(),
        compressed.tx_mode
    );
    // ロスレスの場合は仕様上 tx_mode は必ず ONLY_4X4 (0) になる。
    if header.quantization.lossless {
        assert_eq!(
            compressed.tx_mode,
            0,
            "{}: lossless frame should force tx_mode = ONLY_4X4",
            path.display()
        );
    }
    // 確率値は仕様上 1..=255 の範囲（read_prob/diff_update_prob の性質上 0 にはならない）。
    for &p in compressed.probs.skip_prob.iter() {
        assert!(
            p >= 1,
            "{}: skip_prob should never be 0, got {p}",
            path.display()
        );
    }

    eprintln!(
        "[ok] {}: header_size_in_bytes={header_size}, tx_mode={}, skip_prob={:?}",
        path.display(),
        compressed.tx_mode,
        compressed.probs.skip_prob
    );

    // タイルデータ（圧縮ヘッダの直後から、フレームデータの末尾まで）で decode_tiles を
    // 試みる。トークン復号が未実装のため成功するとは限らないが、パニックしないことを
    // 確認する（Result は Ok/Err のどちらでもよい）。
    let tile_data = &first_frame.data[compressed_end..];
    let mut tile_decoder = TileDecoder::new(&header, &compressed);
    match tile_decoder.decode_tiles(tile_data) {
        Ok(()) => eprintln!("[ok] {}: decode_tiles completed fully", path.display()),
        Err(e) => eprintln!("[info] {}: decode_tiles stopped with {e:?} (expected until token decoding is implemented)", path.display()),
    }
}

#[test]
fn vp90_2_12_droppable_1_compressed_header() {
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00_compressed_header() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}
