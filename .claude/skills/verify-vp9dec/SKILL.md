---
name: verify-vp9dec
description: vp9dec（自作 VP9 デコーダ）の変更・ウェーブを検収するときのチェックリスト。委任先の完了報告を検収する、または自分の変更を確定する前に使う。bit-exact ゲート（公式スイープ両 SIMD 構成・ffmpeg 照合）、[skip] 偽陽性の潰し方、報告数字を鵜呑みにしない照合、静的解析より empirical を信じる原則を定める。
---

# vp9dec 検収チェックリスト

`vp9dec`（Rust 製・ゼロ依存の自作 VP9 デコーダ）。委任先の完了報告や自分の変更を**成果物ベースで**確認する。報告された数字・主張は鵜呑みにせず自分で再実行する。

## 0. 大前提

- **noiria ルートのセッションから作業する場合は必ず `cd .../vp9dec` してから cargo を叩く**。素の `cargo test` はプライマリ dir の noiria ワークスペースを拾う（何度も踏んだ事故）。`cd .../vp9dec && cargo ...` で固定。
- **テスト総数は固定値で assert しない**。ウェーブごとに増減する。毎回実数を記録し、ずれたら `-- --list` で消えた/増えた関数を特定して説明をつける。「0 failed だが数が減った」も放置しない。
- **src/ は外部 crate ゼロ**。テスト dev-dep は可だが self-dep 以外は基本入れない。`git diff Cargo.toml` が空（依存追加なし）を確認。

## 1. 通常スイート

- `cargo test` 全 green。`test result: ok` 行を集計。**conformance_test が既定で走る**（profile 0 の curated 5 + profile 1-3 の 11 を md5 照合）。ここが green なら profile 0-3 の基本適合はガード済み。
- **[skip] 偽陽性を潰す**: conformance / sweep / ffmpeg 照合はベクタや ffmpeg が無いと**黙ってスキップして pass する**。`test result: ok` だけ見て安心しない。`[ok]`/`[xdecode]`/pass 数など「実際に走った証拠」を grep で確認。conformance が `[skip]` ならベクタ未取得 → `bash scripts/fetch-vectors.sh`。

## 2. bit-exact ゲート（デコード経路・SIMD・リファクタに触ったら必須）

**8-bit 出力のバイト完全一致が全変更の不変条件**。u16 化・所有権変更・SIMD・モジュール移動など、どれも 8-bit 出力を 1 バイトも変えてはならない。

- 公式フルスイープ（315/315）を **release で実走**:
  `RUST_MIN_STACK=16777216 cargo test --release --test sweep_test official_vector_sweep -- --nocapture`
  → `total: 315 / pass: 315 / fail: 0`。（debug では自動 skip、部分チェックアウト <300 でも skip。release かつフルコーパス present のときだけ走る。）
- **SIMD / デコード経路の変更は両構成で**: 上記を素で1回、`VP9DEC_NO_SIMD=1` を付けてもう1回。両方 315/315 なら「SIMD 出力 == スカラー出力 == 公式 md5」= bit-exact 証明。
- 「速いが結果が違う」SIMD/最適化は**トレードオフではなくバグ**。1 バイトでも違えば差し戻し。

## 3. ffmpeg 独立照合（合成ベクタの検証／回帰確認）

- `VP9DEC_FFMPEG="<path-to-ffmpeg>" cargo test --test synthetic_seg_test synthetic_streams_cross_decode_against_ffmpeg -- --nocapture`
  → `[xdecode] ... OK` が **8 行**（4 シナリオ × libvpx-vp9 / native vp9）。
- そのパスは**この開発機のローカル shim**（ffmpeg が同梱で PATH 外）。テスト自体は env `VP9DEC_FFMPEG` → PATH の `ffmpeg` 駆動で環境非依存＝CI 移植可能。0 行なら「走っていない」= 検収不合格。

## 4. lint / fmt / docs

- `cargo clippy --all-targets`: ベースライン 3 件（large_enum_variant / identity_op / field_reassign）のみ。**新規警告ゼロ**。
- `cargo fmt --check` clean（tree は正規化済み。素の `cargo fmt` を使ってよい）。
- `docs/implementation-notes.md` に当該変更の節（判断→理由→影響）が追記され、冒頭「Current state index」が最新エントリを指すこと。先送りは `docs/backlog.md` に。

## 原則・落とし穴

- **静的解析より empirical な bit-exact スイープを信じる**。「spec/libvpx 的にこうあるべき」で conformance-passing コードを触るな。実例2件: ループフィルタ triage、`residual.rs` の sub-8x8 chroma index の「4:2:2 バグ」仮説 — 後者は適用したら 4:2:2 ベクタが逆に FAIL し反証。**適合しているコードを推論だけで直さない。必ずスイープで確認**。
- **委任先（Sonnet）は重いスイープをバックグラウンドで待って停止する癖がある**。委任時は「重い release スイープと ffmpeg は待つな、速いゲートだけやって検収に渡せ」と指示し、両スイープ・ffmpeg・速度実測は**自分で確定的に回す**。
- 素の `cargo test` は debug。フルスイープを debug で回すと 315 本が 10 倍遅くなり数分かかる（release-only ガードの理由）。速度計測・スイープは常に `--release`。

## 差し戻し基準

上記いずれか未達、または報告と実測が食い違う場合は原因を特定して委任先へ差し戻すか自分で修正。特に **8-bit 315/315 が両 SIMD 構成で通らない**、**新プロファイル vector が mismatch**、**ffmpeg 8/8 が崩れる**、**[skip] で実は走っていない**は即不合格。
