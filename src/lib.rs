//! vp9dec: 完全自作の VP9 動画デコーダ（依存クレートゼロ）。
//!
//! 参照仕様: VP9 Bitstream & Decoding Process Specification v0.7
//! (Google, 2017年2月22日版, <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>)
//!
//! # マイルストーン
//! - M1: IVF コンテナパーサ、bool デコーダ、非圧縮フレームヘッダのパース
//! - M2: イントラ予測によるキーフレームのデコード
//! - M3: インター予測（動き補償）
//! - M4: コンフォーマンステスト完全通過
//!
//! （モジュールは以降のコミットで段階的に追加する。）

pub mod bit_reader;
pub mod bool_coder;
pub mod header;
pub mod ivf;
pub mod prob_tables;
pub mod quant;
pub mod scan;
pub mod transform;
