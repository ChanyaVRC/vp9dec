//! `tests/vectors/` にある `.ivf` をデコードし、BT.601 で YUV -> RGB 変換したうえで PNG として
//! `target/dump/` に書き出す（目視検収用）。
//!
//! 依存クレートを一切使わない方針（`Cargo.toml` に dependencies を追加しない）に合わせ、
//! PNG エンコードも自前で実装する。zlib の圧縮アルゴリズム（deflate の LZ77 + Huffman）は
//! 実装せず、"stored"（無圧縮）ブロックのみを使う。PNG デコーダはこれでも正しく読める
//! （zlib ストリームとして完全に規格に適合する。圧縮率が悪いだけで、可逆性・正当性には
//! 影響しない）。
//!
//! 実行方法:
//! ```sh
//! # 引数なし: 両ベクタの第 1 フレーム（キーフレーム）を target/dump/<stem>.png に出力する。
//! cargo run --example decode_to_png
//!
//! # 引数あり: 指定ベクタの指定 IVF フレーム番号（0 始まり、デコード順）以降で最初に表示される
//! # フレームを target/dump/<stem>_frame<N>.png に出力する（動き補償ありの中間フレームの
//! # 目視検収用。M3 後半で追加）。
//! cargo run --example decode_to_png -- vp90-2-12-droppable_1 50
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use vp9dec::ivf::IvfReader;
use vp9dec::{decode_keyframe, Decoder, Frame};

fn main() {
    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors");
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("dump");
    fs::create_dir_all(&out_dir).expect("failed to create target/dump");

    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(stem) = args.first() {
        let target_index: usize = args
            .get(1)
            .map(|s| s.parse().expect("フレーム番号は非負整数で指定すること"))
            .unwrap_or(0);
        let path = vectors_dir.join(format!("{stem}.ivf"));
        if !path.exists() {
            eprintln!("[error] テストベクタが見つかりません: {}", path.display());
            std::process::exit(1);
        }
        dump_frame_at(&path, &out_dir, target_index);
        return;
    }

    let vectors = ["vp90-2-12-droppable_1.ivf", "vp90-2-09-subpixel-00.ivf"];

    let mut any_done = false;
    for name in vectors {
        let path = vectors_dir.join(name);
        if !path.exists() {
            eprintln!(
                "[skip] テストベクタが見つかりません: {} (README.md の手順に従って事前にダウンロードしてください)",
                path.display()
            );
            continue;
        }
        any_done = true;
        decode_first_frame(&path, &out_dir);
    }

    if !any_done {
        eprintln!("[warn] デコードできたベクタが 1 つもありませんでした。");
        std::process::exit(1);
    }
}

fn decode_first_frame(path: &Path, out_dir: &Path) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut reader = IvfReader::new(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse IVF {}: {e:?}", path.display()));
    let first_frame = reader
        .next()
        .unwrap_or_else(|| panic!("{} contains no frames", path.display()))
        .unwrap_or_else(|e| panic!("failed to read first frame of {}: {e:?}", path.display()));

    let frame = decode_keyframe(first_frame.data)
        .unwrap_or_else(|e| panic!("decode_keyframe failed for {}: {e:?}", path.display()));

    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let out_path: PathBuf = out_dir.join(format!("{stem}.png"));
    write_png(&frame, &out_path);

    eprintln!("[ok] {} -> {}", path.display(), out_path.display());
}

/// `target_index`（IVF フレーム番号、0 始まり、デコード順）以降で最初に表示される
/// （`show_frame == 1` または `show_existing_frame` で表示される）フレームを 1 枚出力する。
/// `Decoder::decode_frame` はフレーム間状態を要求するため、0 番目から順番にすべて
/// デコードする必要がある（`target_index` だけを単独でデコードすることはできない）。
fn dump_frame_at(path: &Path, out_dir: &Path, target_index: usize) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let reader = IvfReader::new(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse IVF {}: {e:?}", path.display()));

    let mut decoder = Decoder::new();
    let mut result: Option<(usize, Frame)> = None;
    for (i, frame) in reader.enumerate() {
        let frame = frame.unwrap_or_else(|e| {
            panic!("failed to read IVF frame {i} of {}: {e:?}", path.display())
        });
        let outcome = decoder.decode_frame(frame.data).unwrap_or_else(|e| {
            panic!(
                "IVF frame {i} of {} failed to decode: {e:?}",
                path.display()
            )
        });
        if i >= target_index {
            if let Some(decoded) = outcome {
                result = Some((i, decoded));
                break;
            }
        }
    }

    let (actual_index, frame) = result.unwrap_or_else(|| {
        panic!(
            "{}: フレーム {target_index} 以降に表示可能なフレームが見つからなかった",
            path.display()
        )
    });
    if actual_index != target_index {
        eprintln!(
            "[note] {}: フレーム {target_index} は非表示（show_frame==0）だったため、\
             次に表示されるフレーム {actual_index} を出力する。",
            path.display()
        );
    }

    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let out_path: PathBuf = out_dir.join(format!("{stem}_frame{actual_index}.png"));
    write_png(&frame, &out_path);

    eprintln!(
        "[ok] {} (frame {actual_index}) -> {}",
        path.display(),
        out_path.display()
    );
}

fn write_png(frame: &Frame, out_path: &Path) {
    let rgb = yuv_to_rgb_bt601(frame);
    let png_bytes = encode_png(frame.width, frame.height, &rgb);
    fs::write(out_path, &png_bytes)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
}

/// BT.601（limited range, `Y' = 1.164*(Y-16)` 系）で YUV420 を RGB へ変換する。
/// 4:2:0 のクロマは最近傍（`x/2, y/2`）でアップサンプリングする。
fn yuv_to_rgb_bt601(frame: &Frame) -> Vec<u8> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let uv_w = w.div_ceil(2);

    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let y_val = frame.y[y * w + x] as f32;
            let cx = x / 2;
            let cy = y / 2;
            let u_val = frame.u[cy * uv_w + cx] as f32;
            let v_val = frame.v[cy * uv_w + cx] as f32;

            let yy = 1.164 * (y_val - 16.0);
            let cb = u_val - 128.0;
            let cr = v_val - 128.0;

            let r = yy + 1.596 * cr;
            let g = yy - 0.392 * cb - 0.813 * cr;
            let b = yy + 2.017 * cb;

            let idx = (y * w + x) * 3;
            rgb[idx] = clamp_u8(r);
            rgb[idx + 1] = clamp_u8(g);
            rgb[idx + 2] = clamp_u8(b);
        }
    }
    rgb
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// 自前 PNG エンコーダ（IHDR/IDAT/IEND のみ、圧縮なし stored ブロック）。
// ---------------------------------------------------------------------------

const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// 標準の CRC-32（IEEE 802.3、多項式 0xEDB88320）。PNG チャンクの末尾に付与する。
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// zlib (RFC 1950) が要求する Adler-32 チェックサム。
fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// RFC 1951 の "stored"（無圧縮）ブロックのみを使って `data` を deflate 符号化する。
/// stored ブロックは常にバイト境界で始まる（呼び出し側で保証する）ため、
/// 3 ビットのブロックヘッダ（BFINAL + BTYPE=00）はそのまま 1 バイトとして書ける。
fn deflate_stored(data: &[u8]) -> Vec<u8> {
    const MAX_BLOCK: usize = 65535;
    let mut out = Vec::with_capacity(data.len() + data.len() / MAX_BLOCK * 5 + 5);
    if data.is_empty() {
        out.push(1); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(!0u16).to_le_bytes());
        return out;
    }
    let mut offset = 0;
    while offset < data.len() {
        let len = (data.len() - offset).min(MAX_BLOCK);
        let is_final = offset + len == data.len();
        out.push(if is_final { 1 } else { 0 });
        out.extend_from_slice(&(len as u16).to_le_bytes());
        out.extend_from_slice(&(!(len as u16)).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + len]);
        offset += len;
    }
    out
}

/// zlib ストリーム（RFC 1950）: 2 バイトヘッダ + deflate ペイロード + Adler-32。
fn zlib_compress_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 16);
    out.push(0x78); // CMF: CM=8 (deflate), CINFO=7 (32K window)
    out.push(0x01); // FLG: FCHECK が (CMF*256+FLG) % 31 == 0 になるよう選定、FDICT=0
    out.extend_from_slice(&deflate_stored(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// 8bit RGB（truecolor, color type 2）の PNG ファイルを生成する。
/// `rgb` は `width*height*3` バイトの行優先データ。
fn encode_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(
        rgb.len(),
        w * h * 3,
        "rgb データサイズが width*height*3 と一致しない"
    );

    // 各スキャンラインの先頭にフィルタタイプ 0 (None) を付与する。
    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for row in 0..h {
        raw.push(0u8);
        raw.extend_from_slice(&rgb[row * w * 3..(row + 1) * w * 3]);
    }

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // color type: truecolor (RGB)
    ihdr.push(0); // compression method
    ihdr.push(0); // filter method
    ihdr.push(0); // interlace method

    let idat = zlib_compress_stored(&raw);

    let mut out = Vec::with_capacity(PNG_SIGNATURE.len() + ihdr.len() + idat.len() + 64);
    out.extend_from_slice(&PNG_SIGNATURE);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &idat);
    write_chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_known_vector() {
        // "123456789" の CRC-32 (IEEE) は 0xCBF43926 であることが広く知られている。
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn adler32_matches_known_vector() {
        // "Wikipedia" の Adler-32 は 0x11E60398。
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn deflate_stored_roundtrips_via_manual_inflate() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let compressed = deflate_stored(&data);
        let inflated = naive_inflate_stored(&compressed);
        assert_eq!(inflated, data);
    }

    #[test]
    fn encode_png_produces_valid_signature_and_chunks() {
        let rgb = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let png = encode_png(2, 2, &rgb);
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        // IEND チャンクで終わっていること。
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }

    /// テスト専用: stored ブロックのみからなる deflate ストリームを読み戻す
    /// （zlib ヘッダ・adler32 は含まない、`deflate_stored` の出力専用）。
    fn naive_inflate_stored(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut pos = 0;
        loop {
            let header = data[pos];
            let is_final = header & 1 == 1;
            pos += 1;
            let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 4; // LEN + NLEN
            out.extend_from_slice(&data[pos..pos + len]);
            pos += len;
            if is_final {
                break;
            }
        }
        out
    }
}
