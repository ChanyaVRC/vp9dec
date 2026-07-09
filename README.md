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
| M2b | ループフィルタ（デブロッキングフィルタ、仕様 8.8 節）＋公式コンフォーマンス検証 | 完了 |
| M3 前半 | インターフレームのビットストリーム復号（ヘッダ・確率テーブル・モード情報・MV・残差トークン。動き補償の手前まで） | 完了 |
| M3 後半 | 動き補償・サブピクセル補間・参照フレーム管理・確率適応（forward/backward）・全フレーム MD5 コンフォーマンス | **完了** |
| M4 | VP9 コンフォーマンステストベクタの完全通過（手元 2 本以外の公式ベクタへの拡大） | 未着手 |

`decode_keyframe()`（`src/lib.rs`）で実際にキーフレームを最後までデコードし、ループフィルタ
適用済み・表示サイズにクロップ済みの YUV420 `Frame` を得られる。複数フレーム（インター
フレームを含む）を順にデコードする場合は `Decoder::decode_frame()` を使う（後述の「M3 後半」
節参照）。手元の 2 本のテストベクタでは、**すべての表示フレーム**のデコード結果（Y→U→V
連結の I420 バイト列）の MD5 が libvpx 公式配布の `.ivf.md5` と完全一致することを確認済み
（`tests/conformance_test.rs`、詳細は後述。`vp90-2-12-droppable_1`: 99/99 フレーム一致、
`vp90-2-09-subpixel-00`: 20/20 フレーム一致）。

## M1 で実装したもの

- `src/ivf.rs` : IVF コンテナパーサ（32 バイトヘッダ + 12 バイトフレームヘッダの読み取り）
- `src/bool_coder.rs` : VP9 の算術符号（bool coder）デコーダ（仕様 9.2 節）。
  検証用に対になるエンコーダをテスト内に実装し、ラウンドトリップで一致することを確認している。
- `src/header.rs` : 非圧縮フレームヘッダ（uncompressed_header, 仕様 6.2 節）のパース。
  M1 時点ではキーフレームのみサポートしていた（インターフレーム／イントラオンリーフレームは
  M3 前半で対応済み。詳細は後述の「M3 前半」節を参照）。

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

- 8bit（`BitDepth == 8`）のみサポート。`Plane` が `u8` 固定のため、10bit/12bit フレームは
  `decode_keyframe` が `DecodeError::UnsupportedBitDepth` を返す。
- セグメンテーション: `segmentation_enabled == true` のフレームは `TileError::SegmentationNotSupported`
  を返す（既知の制約、M1 から継続）。手元のテストベクタでは両方とも
  `segmentation_enabled == false` であることを確認済み。ループフィルタ（後述）もこれに合わせ、
  `seg_feature_active( SEG_LVL_ALT_L )` を常に偽と仮定して簡略化している。
- 仕様書 9.3.2 節の `partition` の確率選択processの記載についての既知の誤記（erratum）判断は
  M1 から変更なし。詳細は `src/tile.rs` の `read_partition` のコメントを参照。

## M2b（ループフィルタ + 公式コンフォーマンス検証）

以下を実装済み。

- `src/loop_filter.rs`（新規）: デブロッキングフィルタ（仕様 8.8 節）。フレーム全体の走査
  順序（スーパーブロックのラスタ順 →Y/U/V→ 縦エッジ→横エッジ、仕様 8.8 節冒頭の擬似コード）、
  フィルタ強度計算（`build_lvl_lookup`、仕様 8.8.1 節 "Loop filter frame init process"）、エッジ判定（ブロック境界・
  変換ブロック境界・フレーム端の除外、仕様 8.8.2 節）、フィルタサイズ決定（仕様 8.8.3 節）、
  適応フィルタ強度（`limit`/`blimit`/`thresh`、仕様 8.8.4 節）、フィルタ本体（narrow filter =
  4タップ、wide filter = 8/16タップ、flat/flat2 判定を含む、仕様 8.8.5 節）をすべて疑似コード
  どおりの整数演算で実装した。キーフレームのみを対象とするため、`isIntra` は常に真、
  `modeType` は常に 0 に決め打ちしている（`MiInfo` が `ref_frame` を持たないため。M3 引き継ぎ
  メモ参照）。`src/tile.rs::TileDecoder::apply_loop_filter()` から呼び出し、
  `src/lib.rs::decode_keyframe` がタイル復号直後・クロップ前に適用する。
- `src/md5.rs`（新規）: MD5（RFC 1321）の自作実装（依存クレートゼロ方針のため）。
  既知ベクタ（空文字列・`"abc"`・`"message digest"` 等、RFC 1321 に掲載の値）でユニットテスト済み。
- `tests/conformance_test.rs`（新規）: libvpx 公式配布の `.ivf.md5`（`tests/vectors/` に
  ダウンロード、取得手順は後述）と、`decode_keyframe` の最初のキーフレーム出力（Y→U→V 連結の
  I420 バイト列）の MD5 を比較する。**両テストベクタ（`vp90-2-12-droppable_1`・
  `vp90-2-09-subpixel-00`）とも完全一致を確認済み**。特に `vp90-2-09-subpixel-00` は
  キーフレームが疑似乱数状のノイズパターンに見える（`target/dump/*.png` 参照）ため M2 時点では
  デコード結果の正しさに疑義が残っていたが、公式 MD5 と一致したことで正しいデコード結果
  であることが確定した（サブピクセル補間フィルタの効果は後続のインターフレームで初めて
  現れる設計と考えられる。M3 実装後、動き補償ありの中間フレームで改めて目視検収するとよい）。

## M3 前半（インターフレームのビットストリーム復号）

動き補償・サブピクセル補間そのもの（画素生成）を除き、インターフレームのビットストリームを
最初から最後まで正しく読み切れるところまでを実装した。以下を実装済み。

- `src/header.rs` : `uncompressed_header()`（仕様 6.2 節）の非キーフレーム分岐を完全実装。
  `intra_only`/`reset_frame_context`/`refresh_frame_flags`（インター用）/`ref_frame_idx`/
  `ref_frame_sign_bias`/`frame_size_with_refs()`（仕様 6.2.5 節）/`allow_high_precision_mv`/
  `read_interpolation_filter()`（仕様 6.2.7 節）を追加。`frame_size_with_refs()` が参照する
  `RefFrameWidth`/`RefFrameHeight` はフレーム間状態のため、`parse_uncompressed_header()` は
  それを呼び出し側から `&[(u32,u32); NUM_REF_FRAMES]` として受け取る設計にした
  （状態自体は `Decoder`（`src/lib.rs`）が保持する）。インターフレームは `color_config` を
  再送しないため、`Decoder` が直近のキーフレーム/イントラオンリーフレームの値を引き継ぐ。
- `src/prob_tables.rs` : インター関連のツリー（`INTER_MODE_TREE`/`INTERP_FILTER_TREE`/
  `MV_JOINT_TREE`/`MV_CLASS_TREE`/`MV_FR_TREE`）と、仕様 10.5 節から機械抽出したデフォルト
  確率テーブル一式（`DEFAULT_PARTITION_PROBS`/`DEFAULT_Y_MODE_PROBS`/`DEFAULT_UV_MODE_PROBS`/
  `DEFAULT_IS_INTER_PROB`/`DEFAULT_COMP_MODE_PROB`/`DEFAULT_COMP_REF_PROB`/
  `DEFAULT_SINGLE_REF_PROB`/`DEFAULT_INTER_MODE_PROBS`/`DEFAULT_INTERP_FILTER_PROBS`/
  MV 系 8 種）、および仕様 6.5 節 "Motion vector prediction" の定数テーブル
  （`MV_REF_BLOCKS`/`MODE_2_COUNTER`/`COUNTER_TO_CONTEXT`/`IDX_N_COLUMN_TO_SUBBLOCK`/
  `SIZE_GROUP_LOOKUP`）を追加。`mode2txfm_map()` はインターモード値（`NEARESTMV`..`NEWMV`）
  も受け付けるよう拡張（仕様 10.2 節の表は MB_MODE_COUNT=14 全体を定義しており、インター
  モードはすべて `DCT_DCT` に写像される）。
- `src/compressed_header.rs` : `read_inter_mode_probs`/`read_interp_filter_probs`/
  `read_is_inter_probs`/`frame_reference_mode`（`setup_compound_reference_mode` を含む）/
  `frame_reference_mode_probs`/`read_y_mode_probs`/`read_partition_probs`（いずれも非キー
  フレーム版）/`mv_probs`（`update_mv_prob` を含む、仕様 6.3.9〜6.3.18 節）を実装。
  `CompressedHeaderProbs` を「`load_probs`/`save_probs` が操作するすべての確率テーブル」
  （`uv_mode_probs` を除く。更新シンタックスが存在せず常にデフォルト値のため）に拡張し、
  `FrameContext`（= `CompressedHeaderProbs` の別名）と 4 スロットの `FrameContextStore`
  を追加した。`parse_compressed_header_ex()` が新しいインター対応版のエントリポイントで、
  `parse_compressed_header()`（キーフレーム専用、既存 API）はその薄いラッパーとして残した。
- `src/tile.rs` : `inter_frame_mode_info()`（仕様 6.4.11 節）以下を全面実装。
  `read_is_inter`/`intra_block_mode_info`（インターフレーム内のイントラブロック）/
  `read_ref_frames`（`comp_mode`/`comp_ref`/`single_ref_p1`/`single_ref_p2` の文脈導出、
  仕様 9.3.2 節を含む）/`inter_block_mode_info`/`assign_mv`/`read_mv`/`read_mv_component`
  （仕様 6.4.16〜6.4.20 節）を実装。動きベクトル予測（仕様 6.5 節）は `find_mv_refs`/
  `find_best_ref_mvs`/`append_sub8x8_mvs`/`is_inside`/`get_block_mv`/
  `if_same_ref_frame_add_mv`/`if_diff_ref_frame_add_mv` として実装し、純粋な補助計算
  （クランプ・符号反転・しきい値判定）は新設の `src/mv.rs` に切り出した。`UsePrevFrameMvs`
  （仕様 7.2.6 節）にも対応しており、前フレームの `MiGrid`（`Mvs`/`RefFrames` 相当）を
  `TileDecoder::new_with_prev()` 経由で受け取れる。`residual()` は `is_inter` 分岐
  （`predict_inter` の呼び出し位置、`TxType` 決定、`coef_probs` の `is_inter` 添字）と
  `EobTotal`（仕様 6.4.4 節、`is_inter && subsize >= BLOCK_8X8 && EobTotal == 0` で `skip`
  を事後的に 1 にする処理）に対応した。`read_partition` は仕様 9.3.2 節の既知の誤記を修正した
  解釈のまま、`FrameIsIntra` に応じて `KF_PARTITION_PROBS`/`partition_probs` を切り替える。
- `src/predict.rs` : `predict_inter_stub()`（仕様 8.5.2 節のプレースホルダー）を追加。
  動き補償・サブピクセル補間は未実装で、呼び出しても何もしない。仕様 7.4.15 節 NOTE の
  とおり `predict_inter` はシンタックス復号に一切影響しないため、ビットストリームの読了には
  影響しない。
- `src/loop_filter.rs` : `MiInfo` に `ref_frame`/`y_mode`（インター値含む）が揃ったことで、
  `is_intra`/`modeType` の決め打ちを解消し、仕様 8.8.4 節どおり `RefFrames[..][0]`/`YModes`
  の実値を参照するよう修正。**既存のキーフレーム MD5 コンフォーマンステストは引き続き
  全通過を確認済み**（`isIntra`/`modeType` はキーフレームでは従来どおり常に `true`/`0` に
  評価されるため、出力は不変）。
- `src/lib.rs` : フレーム間状態（参照フレームスロットサイズ・フレームコンテキスト 4 スロット・
  `UsePrevFrameMvs` 用の前フレーム `MiGrid`・直近の `color_config`）を保持する `Decoder`
  を新設し、`Decoder::decode_frame()` で 1 フレームずつ順にデコードできるようにした。
  既存の `decode_keyframe()`（第 1 引数がキーフレームであることを検証したうえで、使い捨ての
  `Decoder` を介して `decode_frame()` を呼ぶ）はそのまま維持しており、後方互換。

M3 前半時点での既知の制約（動き補償未実装・確率適応未実装・ループフィルタデルタ未引き継ぎ等）は
すべて次節「M3 後半」で解消した。

## M3 後半（動き補償・確率適応・参照フレーム管理・全フレーム MD5 コンフォーマンス）

M3 前半の引き継ぎ事項をすべて実装し、**両テストベクタで全表示フレームの公式 MD5 完全一致**
（`vp90-2-12-droppable_1`: 99/99、`vp90-2-09-subpixel-00`: 20/20）を達成した。以下を実装済み。

- `src/predict.rs` : `predict_inter()`（仕様 8.5.2 節）。`predict_inter_stub()` を置き換えた。
  内部で仕様どおりの 4 段階の副処理に分割している。
  - `select_mv()`（仕様 8.5.2.1 節）: 4x4 サブブロック単位の MV 選択。クロマかつ
    `MiSize < BLOCK_8X8` の場合は `round_mv_comp_q2`/`round_mv_comp_q4` によるサブサンプリング
    平均化を行う。
  - `clamp_mv_for_plane()`（仕様 8.5.2.2 節）: フレーム端でのクランプ（1/16 pel 精度、
    `INTERP_EXTEND`/`SUBPEL_BITS` を使った境界計算）。
  - `scale_mv_for_plane()`（仕様 8.5.2.3 節）: 参照フレームと現在フレームのサイズが異なる
    場合のスケーリング（`xScale`/`yScale`、`REF_SCALE_SHIFT`）。手元のテストベクタでは
    参照フレームサイズは常に一致していたが、仕様の式をそのまま実装しているため、サイズが
    異なる場合も理論上動作する（未検証）。
  - `block_inter_predict()`（仕様 8.5.2.4 節）: 8 タップ（または `BILINEAR`）サブペル補間
    フィルタによる水平→垂直の 2 パス畳み込み。参照フレーム端の読み出しは
    `Clip3(0, lastX/lastY, ...)` による**クランプ読み**（エッジ拡張ではない）。
  - compound 予測（2 参照の平均、`Round2(pred0+pred1, 1)`）にも対応。
  - フィルタ係数表 `SUBPEL_FILTERS`（`src/prob_tables.rs`）は仕様 8.5.2.4 節の
    `subpel_filters[4][16][8]` を `pdftotext -raw` 抽出テキストからそのまま転記した
    （添字 0..3 が `EIGHTTAP`/`EIGHTTAP_SMOOTH`/`EIGHTTAP_SHARP`/`BILINEAR` の値と一致）。
- `src/dpb.rs`（新規）: `Dpb`/`RefFrameData`。8 スロットの参照フレームバッファ（仕様 8.10 節
  "Reference frame update process"）。`RefFrameData` は表示サイズ（`RefFrameWidth`/`Height`、
  クロマはサブサンプリング後）にクロップ済みのピクセルデータを保持する。これにより
  `predict_inter` のクランプ処理の `lastX`/`lastY` が `Plane::width/height - 1` とそのまま
  一致する。`Decoder::decode_frame()` がタイル復号・ループフィルタ適用後に
  `Dpb::update(refresh_frame_flags, ...)` を呼ぶ。`show_existing_frame` は DPB から該当
  スロットを引いて `Frame` を組み立てて返す（`DecodeError::MissingReferenceFrame` は
  該当スロットが空の場合のみ発生し、適合ビットストリームでは起こらない）。
  参照フレームサイズが現在フレームと異なる場合のスケーリング要否は、手元のテストベクタでは
  常に同一サイズだったため実地では確認できていない（`scale_mv_for_plane` 自体は一般式で
  実装済み）。
- `src/counts.rs`（新規）: 確率適応（仕様 8.4 節）。
  - `Counts`: 仕様 8.3 節 "Clear counts process" が列挙するすべてのカウンタ配列
    （`counts_partition`/`counts_intra_mode`/`counts_uv_mode`/`counts_skip`/
    `counts_is_inter`/`counts_comp_mode`/`counts_comp_ref`/`counts_single_ref`/
    `counts_inter_mode`/`counts_interp_filter`/`counts_tx_size`/`counts_mv_*`/
    `counts_token`/`counts_more_coefs`）。`src/tile.rs` の各シンタックス要素読み取り箇所
    （`read_partition`/`read_skip`/`read_tx_size`/`read_is_inter`/`intra_block_mode_info`/
    `read_ref_frames`/`inter_block_mode_info`/`read_mv`/`read_mv_component`/
    `tokens_and_reconstruct`）に増分処理を追加した。仕様 9.3 節「T 型シンタックス要素は
    ビットを読まずに値が確定する場合でもカウント処理は必ず呼ばれる」という一般規則に従い、
    `partition`（`hasRows`/`hasCols` が両方偽で無条件に `PARTITION_SPLIT` になる場合も含む）
    はすべての分岐でカウントする。
  - `merge_prob`/`merge_probs`（仕様 8.4.1〜8.4.2 節）、`adapt_coef_probs`（仕様 8.4.3 節）、
    `adapt_noncoef_probs`（仕様 8.4.4 節）。
  - **`more_coefs` の "special case"（仕様 9.3.4 節末尾）についての注記**: 仕様 PDF は
    「`more_coefs` については本節末尾で説明する特別な扱いがある」と予告しているが、
    `pdftotext` の `-raw`/`-layout`/デフォルトの 3 モードすべてで該当箇所のテキストが
    実際には印字されていない（v0.7 ドラフト版の欠落と思われる）。本実装では、9.3 節冒頭の
    一般規則（T 型要素は読まなくてもカウントされる）から論理的に導かれる唯一整合的な解釈
    として、`tokens()`（仕様 6.4.24 節）で `checkEob == 0`（直前が `ZERO_TOKEN` だったため
    `more_coefs` を実際には読まない）の場合でも、`counts_more_coefs[...][1]` を値 1
    （「続きがある」）として加算する実装を採用した。この解釈は両テストベクタの
    **全フレーム MD5 完全一致**によって実証的に裏付けられている
    （`src/tile.rs::tokens_and_reconstruct` のコメント参照）。
  - `Decoder::decode_frame()` の `refresh_probs()` 相当処理（仕様 6.1.2 節・7.1.2 節）:
    `load_probs(ctx)` は `tx_probs`/`skip_prob` **を除く**すべてのテーブルを forward update
    前の値（`starting_probs`）へ戻すことを指す点に注意（`compressed_header()` 側は
    `tx_probs`/`skip_prob` も含めてすべて forward update するが、`refresh_probs()` の
    `load_probs` はその 2 つだけ forward update 後の値のまま残す）。`adapt_coef_probs()` は
    その状態の `coef_probs` に対して適用され、`FrameIsIntra == 0` の場合のみ `load_probs2(ctx)`
    （`tx_probs`/`skip_prob` も forward update 前へ戻す）→ `adapt_noncoef_probs()` が続く。
    この「forward update 前の値に戻してから backward adaptation する」という 2 段階の
    構造を誤解して forward update 後の値に対して adaptation してしまうと、キーフレーム単発
    では問題が顕在化しない（backward adaptation 自体が発火しないため）が、複数フレームに
    またがる MD5 コンフォーマンスで確実に失敗する。本実装は `Decoder::decode_frame()` 内で
    `starting_probs`（forward update 前）を保持しておき、上記の手順どおりに実装している。
  - `uv_mode_probs` は `compressed_header()` に forward update シンタックスが存在しない
    （常にビットストリームでは更新されない）が、仕様 8.4.4 節の `adapt_noncoef_probs()` は
    `uv_mode_probs` も backward adaptation の対象に含めている。M3 前半では
    `CompressedHeaderProbs` にこのフィールドがなく、インターフレーム内のイントラブロックの
    `uv_mode` 読み取りは常に `DEFAULT_UV_MODE_PROBS` 固定値を使っていた（誤り）。
    `CompressedHeaderProbs::uv_mode_probs` フィールドを追加し、`load_probs`/`save_probs`/
    backward adaptation の対象に含めるよう修正した。
- `src/header.rs` : ループフィルタの `ref_deltas`/`mode_deltas`（仕様 7.2 節、
  `setup_past_independence()` 時のみリセットされフレーム間で持続する状態）を
  `parse_uncompressed_header()`/`parse_loop_filter_params()` の引数として前フレームの値を
  受け取れるように変更した（`ref_frame_sizes` と同じ設計パターン）。`Decoder` がこの状態を
  保持し、フレームごとに更新する。
- `src/lib.rs` : `Decoder` に DPB（`dpb: Dpb`）・ループフィルタデルタ（`loop_filter_deltas`）・
  `LastFrameType`（`adapt_coef_probs` の `updateFactor` 計算に使う、仕様 8.4.3 節）を追加。
  **公開 API を変更**した: `Decoder::decode_frame(&mut self, data: &[u8]) -> Result<Option<Frame>, DecodeError>`
  （`DecodeOutcome` 列挙体は廃止）。`None` は「非表示フレーム」（`show_frame == 0`、いわゆる
  droppable/altref フレーム）を表す。`show_existing_frame` フレームは DPB から実際に
  ピクセルデータを引いて `Some(Frame)` を返すようになった（M3 前半では
  `DecodeOutcome::ShowExisting { frame_to_show_map_idx }` を返すのみで実データを持たなかった）。
  `decode_keyframe()`（第 1 引数がキーフレームであることを検証したうえで内部的に `Decoder`
  経由で呼ぶ）は引き続き後方互換の単発 API として残している。
- `reset_frame_context == 2`（該当フレームコンテキストのみリセット）は引き続き未実装
  （`FrameIsIntra || error_resilient_mode` の場合は常に全 4 スロットをリセット）。
  `frame_context_idx` はこの場合いずれも 0 に固定されるため、ビットストリーム読み取りには
  影響しない（M3 前半からの既知の制約を継続）。
- セグメンテーション未対応は継続（M1 からの既知の制約）。

### デバッグ時の切り分け経過（参考）

実装は 1 回目の全フレーム MD5 テストで両ベクタとも完全一致したため、大掛かりな切り分けは
不要だった。ただし実装前に想定していた切り分け方針（不一致が出た場合用）は以下のとおりで、
今後 M4 で新しいテストベクタを追加する際にも有効なはずなので記録しておく。
1. 第 2 フレーム（最初のインターフレーム）だけを比較 → 補間フィルタ・DPB・MV 予測など
   「1 フレーム内で閉じる」バグかどうかを切り分ける。
2. 確率適応を無効化（`adapt_coef_probs`/`adapt_noncoef_probs` の呼び出しをコメントアウトし、
   forward update 後の値をそのまま保存する M3 前半相当の挙動に戻す）した場合に何フレーム目
   まで一致するかを見る → 一致するフレーム数が伸びれば適応ロジックのバグ、変わらなければ
   動き補償・DPB 側のバグと判断できる。
3. 予測画素のみ（残差を強制的にゼロにする）で比較 → `predict_inter` 自体の正しさと
   トークン復号・逆量子化・逆変換の正しさを分離して検証できる。

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
`tests/conformance_test.rs`（M2b で追加、M3 後半で拡張）は 2 種類の検証を行う。
- `*_first_keyframe_matches_official_md5`（M2b）: `decode_keyframe()` の最初のキーフレーム
  出力（Y→U→V 連結の I420 バイト列）の MD5 が libvpx 公式配布の `.ivf.md5` の 1 行目と
  完全一致することを検証する。
- `*_all_frames_match_official_md5`（M3 後半で追加）: `Decoder::decode_frame()` で
  **全 IVF フレーム**を順にデコードし、表示される（`Some(Frame)` が返る）フレームすべてに
  ついて、`.ivf.md5` の対応する行と MD5 が完全一致することを検証する（動き補償・確率適応・
  DPB・ループフィルタのフレーム間状態がすべて正しくないと通らない）。
  `vp90-2-12-droppable_1`: 99/99 出力フレーム、`vp90-2-09-subpixel-00`: 20/20 出力フレームで
  完全一致を確認済み。

`tests/inter_frame_test.rs`（M3 前半で追加）は `Decoder::decode_frame()` を使い、
各テストベクタの**全 IVF フレーム**（キーフレーム・インターフレーム・`vp90-2-12-droppable_1`
の droppable フレームを含む）を順にデコードし、`uncompressed_header`＋`compressed_header`＋
全タイルのモード情報・MV・残差トークンをパニックなく最後まで読み切れることを検証する
（`vp90-2-09-subpixel-00` は 20 フレーム、`vp90-2-12-droppable_1` は 99 フレームすべてを
確認済み）。画素の正しさは `conformance_test.rs` 側で検証するため、ここではビットストリーム
読了のみを見る。テストベクタ・MD5 ファイルはリポジトリに含めていない（`.gitignore` 対象）
ため、以下の手順で事前にダウンロードしておく必要がある。ダウンロードしていない場合、該当
テストは早期 return + `eprintln!` でスキップされ、テストスイート全体は失敗しない。

```sh
mkdir -p tests/vectors
curl -o tests/vectors/vp90-2-12-droppable_1.ivf \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-12-droppable_1.ivf
curl -o tests/vectors/vp90-2-09-subpixel-00.ivf \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-09-subpixel-00.ivf
curl -o tests/vectors/vp90-2-12-droppable_1.ivf.md5 \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-12-droppable_1.ivf.md5
curl -o tests/vectors/vp90-2-09-subpixel-00.ivf.md5 \
  https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp90-2-09-subpixel-00.ivf.md5
```

（PowerShell の場合は `Invoke-WebRequest -Uri <URL> -OutFile <path>` を使用する。）

`.ivf.md5` は `md5sum` 互換フォーマット（`<32文字hex>␠␠<ファイル名>`）で、1 行 = 1 出力
（表示）フレーム（Y→U→V 連結の I420 バイト列）の MD5 を記録している。`show_frame == 0` の
非表示フレームは行を持たない。`*_first_keyframe_matches_official_md5` は 1 行目のみ、
`*_all_frames_match_official_md5` は全行を使用する。

テストベクタの一覧は libvpx リポジトリの
[`test/test-data.sha1`](https://github.com/webmproject/libvpx/blob/main/test/test-data.sha1) に
記載されている。`vp90-2-*.ivf`（`invalid-` プレフィックスが付かないもの）が生の IVF コンテナで、
それ以外は WebM コンテナ (.webm) で提供されている。

### PNG ダンプ（目視検収用）

`examples/decode_to_png.rs` は `.ivf` をデコードし、BT.601（limited range）で YUV → RGB
変換したうえで `target/dump/` に PNG として書き出す。PNG エンコードも依存クレートを使わず
自前実装している（zlib は無圧縮の "stored" ブロックのみ使用、CRC-32/Adler-32 も自前実装）。

引数なしの場合は両ベクタの第 1 フレーム（キーフレーム）を `target/dump/<ベクタ名>.png` に
書き出す（M2b からの既存動作）。

```sh
cargo run --example decode_to_png
```

引数でベクタ名（拡張子なし）と IVF フレーム番号（0 始まり、デコード順）を指定すると、
そのフレーム以降で最初に表示されるフレームを `target/dump/<ベクタ名>_frame<N>.png` に
書き出す（M3 後半で追加。動き補償ありの中間フレームの目視検収用。`Decoder` はフレーム間状態を
要求するため、内部では 0 番目から指定フレームまで順にデコードする）。

```sh
# vp90-2-12-droppable_1 の第 50 フレーム前後をダンプする例
cargo run --example decode_to_png -- vp90-2-12-droppable_1 50
```

`vp90-2-12-droppable_1` の第 50 フレーム付近（建設現場の実写映像）で目視検収済み。
`vp90-2-09-subpixel-00` は仕様上どのフレームも疑似乱数状のノイズパターンになる合成
テストベクタのため、動き補償後も見た目はノイズのまま変わらない（MD5 一致で正しさは別途
確認済み。「M2b」節の既存の注記も参照）。

## ライセンス

MIT
