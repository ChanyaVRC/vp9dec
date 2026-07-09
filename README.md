# vp9dec

完全自作の VP9 動画デコーダ（Rust, 依存クレートゼロ）。

## 目的

ノベルゲームエンジン [Noiria](../noiria) への統合を見据え、VP9（ロイヤリティフリーの動画コーデック）の
デコーダをクリーンルームで実装する。外部クレートには依存せず（dev-dependencies も含めゼロ依存）、
Rust 標準ライブラリのみで実装する。

一次情報として [VP9 Bitstream & Decoding Process Specification v0.7](
https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf)
（Google, 2017年2月22日版）を参照する。既存 OSS 実装（libvpx 等）のソースコードは参照しない
（クリーンルーム実装）。

## 段階計画

| マイルストーン | 内容 | 状態 |
| --- | --- | --- |
| M1 | コンテナ (IVF) パーサ、bool デコーダ、非圧縮フレームヘッダのパース | 完了 |
| M2 | イントラ予測によるキーフレームのデコード（圧縮ヘッダ・タイル・変換・量子化・ループフィルタ） | 未着手 |
| M3 | インター予測（動き補償）によるフレーム間デコード | 未着手 |
| M4 | VP9 コンフォーマンステストベクタの完全通過 | 未着手 |

## M1 で実装したもの

- `src/ivf.rs` : IVF コンテナパーサ（32 バイトヘッダ + 12 バイトフレームヘッダの読み取り）
- `src/bool_coder.rs` : VP9 の算術符号（bool coder）デコーダ（仕様 9.2 節）。
  検証用に対になるエンコーダをテスト内に実装し、ラウンドトリップで一致することを確認している。
- `src/header.rs` : 非圧縮フレームヘッダ（uncompressed_header, 仕様 6.2 節）のパース。
  キーフレームのみサポートし、インターフレーム／イントラオンリーフレームは M2 以降で対応する。

## テスト

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
```

### 実データによる検証

`tests/header_test.rs` は WebM 公式のテストベクタ（libvpx コンフォーマンステスト用データ）を使って
IVF パーサとヘッダパーサを検証する。テストベクタはリポジトリに含めていない（`.gitignore` 対象）ため、
以下の手順で事前にダウンロードしておく必要がある。ダウンロードしていない場合、該当テストは
早期 return + `eprintln!` でスキップされ、テストスイート全体は失敗しない。

```sh
mkdir -p tests/vectors
curl -o tests/vectors/vp90-2-12-droppable_1.ivf \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-12-droppable_1.ivf
curl -o tests/vectors/vp90-2-09-subpixel-00.ivf \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-09-subpixel-00.ivf
```

（PowerShell の場合は `Invoke-WebRequest -Uri <URL> -OutFile <path>` を使用する。）

テストベクタの一覧は libvpx リポジトリの
[`test/test-data.sha1`](https://github.com/webmproject/libvpx/blob/main/test/test-data.sha1) に
記載されている。`vp90-2-*.ivf`（`invalid-` プレフィックスが付かないもの）が生の IVF コンテナで、
それ以外は WebM コンテナ (.webm) で提供されている。

## ライセンス

MIT
