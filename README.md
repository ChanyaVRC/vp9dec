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
| M2 | イントラ予測によるキーフレームのデコード（圧縮ヘッダ・タイル・変換・量子化・ループフィルタ） | 進行中 |
| M3 | インター予測（動き補償）によるフレーム間デコード | 未着手 |
| M4 | VP9 コンフォーマンステストベクタの完全通過 | 未着手 |

## M1 で実装したもの

- `src/ivf.rs` : IVF コンテナパーサ（32 バイトヘッダ + 12 バイトフレームヘッダの読み取り）
- `src/bool_coder.rs` : VP9 の算術符号（bool coder）デコーダ（仕様 9.2 節）。
  検証用に対になるエンコーダをテスト内に実装し、ラウンドトリップで一致することを確認している。
- `src/header.rs` : 非圧縮フレームヘッダ（uncompressed_header, 仕様 6.2 節）のパース。
  キーフレームのみサポートし、インターフレーム／イントラオンリーフレームは M2 以降で対応する。

## M2 進捗（キーフレームの compressed_header・タイル・モード情報の復号）

以下を実装済み（すべてキーフレーム / `FrameIsIntra == 1` のみが対象。インター予測関連の
シンタックスは仕様上そもそも読まれないため未実装）。

- `src/bool_coder.rs` : `read_tree`（仕様 9.3.3 節 "Tree decoding process"）を追加。
  既存の `read_bool`/`read_literal`/`exit_bool` の API は変更していない。
- `src/prob_tables.rs`（新規）: ツリー定義（`PARTITION_TREE` 等）とデフォルト確率テーブル
  （`KF_PARTITION_PROBS`、`KF_Y_MODE_PROBS`、`KF_UV_MODE_PROBS` は仕様 10.4 節、
  `DEFAULT_COEF_PROBS`、`DEFAULT_TX_PROBS`、`DEFAULT_SKIP_PROB`、`INV_MAP_TABLE` は仕様
  10.5 節・6.3.5 節、ブロックサイズ変換テーブル群は仕様 10.2 節から転記）。
  `DEFAULT_COEF_PROBS`（576 通りの `[u8; 3]`、計 1728 個の数値）は転記ミスを避けるため、
  仕様書 PDF から機械的に抽出した数値列と手作業で書いたテーブルが完全一致することを
  スクリプトで検証している。
- `src/compressed_header.rs`（新規）: `compressed_header()`（仕様 6.3 節）。`read_tx_mode`
  （ロスレス時は `ONLY_4X4` 固定）、`diff_update_prob`/`decode_term_subexp`/`inv_remap_prob`
  による `tx_probs`/`coef_probs`/`skip_prob` の更新を実装。`FrameIsIntra == 0` でのみ呼ばれる
  `read_inter_mode_probs()` 以降は「読み飛ばす」のではなく「そもそも呼ばれない」ため未実装。
- `src/tile.rs`（新規）: `decode_tiles`/`decode_tile`/`decode_partition`/`decode_block`/
  `intra_frame_mode_info`（仕様 6.4 節）。`MiInfo`/`MiGrid` でフレーム全体の mode info を
  8x8 単位のグリッドとして保持し、`AbovePartitionContext`/`LeftPartitionContext` も実装。

### 設計判断: `TileError::ResidualNotImplemented`

係数（トークン）の復号（仕様 6.4.24〜6.4.26 節、Pareto テーブルや coefband を使う部分）は
別タスクで実装するため、`residual()`（仕様 6.4.21 節）はスタブとして置いている。仕様上
`residual()` は `skip == 1` の場合ビットストリームから一切読まないため
（`if ( !skip ) { nonzero = tokens( ... ) ... }`）、skip ブロックのみで構成される区間は
正しく処理できる。`skip == 0` のブロックでトークン位置に到達した場合は
`TileError::ResidualNotImplemented` を返し、呼び出し側（統合テスト）はこのエラーを
「未実装機能に到達しただけで、それより前の処理は正しい」という意味で許容する。
実データ（`vp90-2-12-droppable_1.ivf`）で `decode_tiles` を試したところ、パニックせず
`ResidualNotImplemented` に到達することを確認済みで、少なくとも最初の非 skip ブロックまでの
partition・mode_info の復号ロジックが機能していることを示している。

### 既知の制約

- セグメンテーション: `segmentation_enabled == true` のフレームは `TileError::SegmentationNotSupported`
  を返す。`src/header.rs` が `segmentation_update_map` 等の詳細パラメータをまだ保持していないため。
  手元のテストベクタでは両方とも `segmentation_enabled == false` であることを確認済み。
- 仕様書 9.3.2 節の `partition` の確率選択process には「`FrameIsIntra == 0` のとき
  `kf_partition_probs` を使う」という記載があるが、これは既知の誤記（erratum）と判断した。
  `compressed_header()`（仕様 6.3 節）は `partition_probs` を `FrameIsIntra == 0` の場合にしか
  読み込まないため、文面通りに実装すると `FrameIsIntra == 1`（キーフレーム）で未更新のままの
  `partition_probs` を参照することになり、`kf_` 接頭辞の意図（キーフレーム専用の固定表）とも
  矛盾する。本実装ではキーフレームで常に `KF_PARTITION_PROBS` を使う（`src/tile.rs` の
  `read_partition` にコメントで詳細を記載）。

### 次の統合タスクへの引き継ぎ

- トークン復号（`tokens()`/`read_coef()`、Pareto テーブル 10.3 節）、イントラ予測
  （`predict_intra()`、仕様 8.5.1 節）、逆量子化・逆変換との結合（`src/quant.rs`/
  `src/transform.rs`、他タスクで実装済み）、再構成（`reconstruct()`）を
  `src/tile.rs::TileDecoder::read_residual` の中に実装していくことになる。
  その際 `AboveNonzeroContext`/`LeftNonzeroContext` の追加が必要（本実装では未使用）。
  また `get_uv_tx_size()`/`get_plane_block_size()`（仕様 6.4.22〜6.4.23 節、`ss_size_lookup`）も
  まだ転記していない。
- `MiGrid`/`above_partition_context`/`left_partition_context` は `TileDecoder` の非公開フィールド
  だが、`pub fn mi_grid(&self)` で読み取り専用アクセスができる。

## テスト

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
```

### 実データによる検証

`tests/header_test.rs` は WebM 公式のテストベクタ（libvpx コンフォーマンステスト用データ）を使って
IVF パーサとヘッダパーサを検証する。`tests/compressed_header_test.rs` は同じテストベクタを使い、
最初のキーフレームについて `compressed_header` の読了と `decode_tiles`（パニックしないことのみ）を
検証する。テストベクタはリポジトリに含めていない（`.gitignore` 対象）ため、
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
