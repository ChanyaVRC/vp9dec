//! WebM 公式テストベクタ（libvpx コンフォーマンステスト用データ）を使った統合テスト。
//!
//! `tests/vectors/` にダウンロード済みの `.ivf` ファイルがあれば、以下を検証する:
//! - IVF コンテナが正しくパースできる
//! - 第 1 フレームがキーフレームである
//! - 非圧縮ヘッダの width/height が IVF コンテナヘッダの width/height と一致する
//!
//! テストベクタが存在しない環境（例: ネットワークアクセスができない CI）でもテストスイート
//! 全体が失敗しないよう、ファイルが見つからない場合は早期 return + `eprintln!` でそのテストを
//! スキップする。取得方法は README.md を参照。

use std::path::Path;

use vp9dec::header::{parse_uncompressed_header, FrameHeader, FrameType, NUM_REF_FRAMES};
use vp9dec::ivf::IvfReader;

const NO_REF_SIZES: [(u32, u32); NUM_REF_FRAMES] = [(0, 0); NUM_REF_FRAMES];
const NO_LF_DELTAS: ([i8; 4], [i8; 2]) = ([1, 0, -1, -1], [0, 0]);

/// 指定したテストベクタで「IVF が読める / 第 1 フレームがキーフレーム /
/// ヘッダの width・height が IVF ヘッダと一致する」ことを検証する。
/// ファイルが存在しない場合は早期 return する。
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
    let ivf_header = reader.header().clone();
    assert_eq!(&ivf_header.fourcc, b"VP90", "codec fourcc should be VP90");

    let mut frames = reader;
    let first_frame = frames
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let (parsed, _consumed) =
        parse_uncompressed_header(first_frame.data, &NO_REF_SIZES, NO_LF_DELTAS).unwrap_or_else(
            |e| {
                panic!(
                    "failed to parse uncompressed header of first frame in {}: {e:?}",
                    path.display()
                )
            },
        );

    match parsed {
        FrameHeader::New(f) => {
            assert_eq!(
                f.frame_type,
                FrameType::KeyFrame,
                "first frame of {} should be a key frame",
                path.display()
            );
            assert_eq!(
                f.width,
                ivf_header.width as u32,
                "decoded width should match IVF container header for {}",
                path.display()
            );
            assert_eq!(
                f.height,
                ivf_header.height as u32,
                "decoded height should match IVF container header for {}",
                path.display()
            );
            eprintln!(
                "[ok] {}: {}x{}, frame_type={:?}",
                path.display(),
                f.width,
                f.height,
                f.frame_type
            );
        }
        FrameHeader::ShowExistingFrame { .. } => {
            panic!(
                "first frame of {} unexpectedly used show_existing_frame",
                path.display()
            );
        }
    }
}

#[test]
fn vp90_2_12_droppable_1() {
    check_vector("vp90-2-12-droppable_1.ivf");
}

#[test]
fn vp90_2_09_subpixel_00() {
    check_vector("vp90-2-09-subpixel-00.ivf");
}
