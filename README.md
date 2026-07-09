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
| M2 | イントラ予測によるキーフレームのデコード（圧縮ヘッダ・タイル・トークン復号・変換・量子化・再構成） | 完了（ループフィルタを除く） |
| M2b | ループフィルタ（デブロッキングフィルタ、仕様 8.8 節） | 未着手 |
| M3 | インター予測（動き補償）によるフレーム間デコード | 未着手 |
| M4 | VP9 コンフォーマンステストベクタの完全通過 | 未着手 |

`decode_keyframe()`（`src/lib.rs`）で実際にキーフレームを最後までデコードし、YUV420 の
`Frame` を得られる。ただしループフィルタ（M2b）が未適用のため、ブロック境界に軽微な
ノイズが残る場合がある。

## M1 で実装したもの

- `src/ivf.rs` : IVF コンテナパーサ（32 バイトヘッダ + 12 バイトフレームヘッダの読み取り）
- `src/bool_coder.rs` : VP9 の算術符号（bool coder）デコーダ（仕様 9.2 節）。
  検証用に対になるエンコーダをテスト内に実装し、ラウンドトリップで一致することを確認している。
- `src/header.rs` : 非圧縮フレームヘッダ（uncompressed_header, 仕様 6.2 節）のパース。
  キーフレームのみサポートし、インターフレーム／イントラオンリーフレームは M2 以降で対応する。

## M2（キーフレームのイントラ復号）

以下を実装済み（すべてキーフレーム / `FrameIsIntra == 1` のみが対象。インター予測関連の
シンタックスは仕様上そもそも読まれないため未実装）。

- `src/bool_coder.rs` : `read_tree`（仕様 9.3.3 節 "Tree decoding process"）。
- `src/prob_tables.rs` : ツリー定義（`PARTITION_TREE`/`TOKEN_TREE`/`INTRA_MODE_TREE` 等）と
  デフォルト確率テーブル一式。`KF_PARTITION_PROBS`/`KF_Y_MODE_PROBS`/`KF_UV_MODE_PROBS` は
  仕様 10.4 節、`DEFAULT_COEF_PROBS`/`DEFAULT_TX_PROBS`/`DEFAULT_SKIP_PROB`/`INV_MAP_TABLE` は
  仕様 10.5 節・6.3.5 節、ブロックサイズ変換テーブル群・`SS_SIZE_LOOKUP` は仕様 10.2 節・
  6.4.23 節、`PARETO_TABLE`（128x8）は仕様 10.3 節、`COEFBAND_4X4`/`coefband_8x8plus()`/
  `ENERGY_CLASS`/`EXTRA_BITS`/`CAT_PROBS`/`mode2txfm_map()` は仕様 6.4.24〜6.4.26 節から転記。
  数値の大きいテーブル（`DEFAULT_COEF_PROBS`、`PARETO_TABLE`）は手作業の転記ミスを避けるため、
  `pdftotext -layout` で抽出した仕様 PDF のテキストから `grep -oE` による正規表現で数値列を
  機械的に抽出し、そのまま Rust の配列リテラルへ変換して埋め込んでいる
  （`coefband_8x8plus` も同様の手順で抽出した 1024 要素の実データから、末尾 1003 要素がすべて
  `5` であることを確認したうえで、配列ではなく関数として圧縮実装した）。
- `src/compressed_header.rs` : `compressed_header()`（仕様 6.3 節）。
- `src/tile.rs` : `decode_tiles`/`decode_tile`/`decode_partition`/`decode_block`/
  `intra_frame_mode_info`（仕様 6.4 節）に加え、本タスクで **`residual()`（仕様 6.4.21 節）を
  完全実装**した。プレーンごとに以下を行う:
  1. `get_uv_tx_size()`/`get_plane_block_size()`（仕様 6.4.22〜6.4.23 節）でクロマの変換
     サイズ・ブロックサイズを決定。
  2. `predict_intra()`（`src/predict.rs`、仕様 8.5.1 節）でイントラ予測。`skip` の値に関わらず
     必ず実行される（仕様どおり、predict は `!skip` の外側）。
  3. `skip == 0` の場合のみ `tokens_and_reconstruct()` でトークン復号
     （`tokens()`、仕様 6.4.24 節）→ `get_scan()`/`TxType` 決定（仕様 6.4.25 節）→
     逆量子化・逆変換・再構成（`reconstruct()`、仕様 8.6.2 節、`src/quant.rs`/`src/transform.rs`
     を利用）を行う。
  4. `AboveNonzeroContext`/`LeftNonzeroContext` を更新（仕様 6.4.21 節末尾のループ。
     `skip`/フレーム端で読まなかった場合も含め、必ず `nonzero` の値で更新する）。
- `src/framebuffer.rs`（新規）: `Plane`（`CurrFrame[plane]` 相当）。フレームバッファは
  スーパーブロック境界（`Sb64Cols*64`/`Sb64Rows*64`、クロマはサブサンプリング後）まで
  切り上げて確保する。理由: `predict_intra`/`reconstruct` の書き込みは `(MiCols*8, MiRows*8)`
  をわずかに超える場合があるため（読み出し側は `Min(maxX, ...)` で必ずクリップされるが、
  書き込み側の `pred[i][j]`/`Dequant[i][j]` の代入はクリップされない）。
- `src/predict.rs`（新規）: `predict_intra()`（仕様 8.5.1 節）。VP9 の 10 イントラモード
  （`DC`/`V`/`H`/`D45`/`D135`/`D117`/`D153`/`D207`/`D63`/`TM`、smooth 系フィルタは VP9 に
  存在しないため実装していない）をすべて実装。`aboveRow`/`leftCol` の可用性判定・
  フレーム端でのクランプ・`notOnRight`（4x4 変換時のみ右側参照を許可）を仕様どおりに扱う。
- `src/lib.rs` : 公開 API `decode_keyframe(frame_data: &[u8]) -> Result<Frame, DecodeError>`。
  非圧縮ヘッダ→圧縮ヘッダ→タイル復号を一気通貫で行い、`Frame { width, height, y, u, v }`
  （表示サイズにクロップ済み、仕様 8.9 節の出力プロセスに準拠）を返す。

### 既知の制約

- **ループフィルタ未実装（M2b）**: デブロッキングフィルタ（仕様 8.8 節）は未実装のため、
  出力画像にはブロック境界の軽微なノイズが残る場合がある。`decode_keyframe` はこれを
  補正しない。
- 8bit（`BitDepth == 8`）のみサポート。`Plane` が `u8` 固定のため、10bit/12bit フレームは
  `decode_keyframe` が `DecodeError::UnsupportedBitDepth` を返す。
- セグメンテーション: `segmentation_enabled == true` のフレームは `TileError::SegmentationNotSupported`
  を返す（既知の制約、M1 から継続）。手元のテストベクタでは両方とも
  `segmentation_enabled == false` であることを確認済み。
- 仕様書 9.3.2 節の `partition` の確率選択processの記載についての既知の誤記（erratum）判断は
  M1 から変更なし。詳細は `src/tile.rs` の `read_partition` のコメントを参照。

### M2b（ループフィルタ）への引き継ぎメモ

- 実装箇所の候補: `src/lib.rs::decode_keyframe` の中で `decoder.decode_tiles(...)` の直後、
  `decoder.planes()` を読む前に `loop_filter_frame()`（仕様 8.8.1 節）相当の処理を挟む形になる。
  `TileDecoder` に `pub fn planes_mut(&mut self) -> &mut [Plane; 3]`
  （または専用のループフィルタ関数に `&mut self` を渡す形）を追加する必要がある。
- ループフィルタの強度・要否判定には `MiGrid`（`tx_size`/`skip`/`mi_size`/`y_mode` 等）と
  `header.loop_filter`（`level`/`sharpness`/`ref_deltas`/`mode_deltas`）の両方が必要。
  `MiGrid` は既に `TileDecoder::mi_grid()` で読み取り専用アクセス可能。`loop_filter` パラメータは
  `NewFrameHeader.loop_filter` に保持済み（`src/header.rs`）だが `TileDecoder` はまだ保持して
  いないので、`TileDecoder::new` に渡すか、フィルタ処理を `TileDecoder` の外（`lib.rs`）に
  独立した関数として置き `&NewFrameHeader` を直接渡すほうが自然。
- `filter_level == 0` の場合は仕様上フィルタ処理自体をスキップしてよい（両テストベクタで
  実際の値を確認しておくこと）。

### M3（インター予測）への引き継ぎメモ

- `src/header.rs` はキーフレーム（`frame_type == KEY_FRAME`）のみ対応。インターフレームの
  `uncompressed_header` 追加フィールド（`ref_frame_idx`、`ref_frame_sign_bias`、
  `allow_high_precision_mv`、`interpolation_filter`、`frame_size_with_refs` 等）が未パース。
- `src/compressed_header.rs` の `read_inter_mode_probs()` 以降（`inter_mode_probs`/
  `interp_filter_probs`/`is_inter_prob`/`frame_reference_mode`/`comp_mode_probs`/
  `single_ref_prob`/`comp_ref_prob`/`y_mode_probs`（非キーフレーム版）/`partition_probs`
  （非キーフレーム版）/`mv_probs`）が未実装。`FrameIsIntra == 0` でのみ呼ばれる。
- `src/tile.rs` の `mode_info()` は `intra_frame_mode_info()` のみ。`inter_frame_mode_info()`
  （仕様 6.4.11〜6.4.20 節、`find_mv_refs`/MV 予測・復号を含む）が未実装。
- `src/predict.rs` に `predict_inter()`（仕様 8.5.2 節、動き補償・サブピクセル補間フィルタ）を
  追加する必要がある。`vp90-2-09-subpixel-00.ivf` はサブピクセル補間フィルタのテスト用ベクタ
  （名前のとおり）で、キーフレーム自体は疑似乱数的なノイズパターンであることを
  `examples/decode_to_png.rs` の出力で確認済み（後続のインターフレームで初めてサブピクセル
  補間の効果が現れる設計と推測される。M3 実装後、動き補償ありの中間フレームで改めて
  目視検収するとよい）。
- `MiGrid`/`above_partition_context`/`left_partition_context`/`above_nonzero_context`/
  `left_nonzero_context` は `TileDecoder` の非公開フィールドだが、`pub fn mi_grid(&self)`/
  `pub fn planes(&self)` で読み取り専用アクセスができる。

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
検証する。`tests/decode_test.rs` は `decode_keyframe()`（公開 API）で最初のキーフレームを
最後まで完全にデコードし、Y プレーンの統計値（分散が 0 でない・全ピクセル同値でない・
`min < 50 && max > 200`）で実写系ベクタとして妥当な出力であることを検証する。
テストベクタはリポジトリに含めていない（`.gitignore` 対象）ため、
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

### PNG ダンプ（目視検収用）

`examples/decode_to_png.rs` は `tests/vectors/` の各 `.ivf` の第 1 フレームをデコードし、
BT.601（limited range）で YUV → RGB 変換したうえで `target/dump/<ベクタ名>.png` に書き出す。
PNG エンコードも依存クレートを使わず自前実装している（zlib は無圧縮の "stored" ブロックのみ
使用、CRC-32/Adler-32 も自前実装）。

```sh
cargo run --example decode_to_png
```

## ライセンス

MIT
