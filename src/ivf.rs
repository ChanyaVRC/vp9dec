//! IVF コンテナパーサ。
//!
//! IVF は libvpx (VP8/VP9) のテストや配布でよく使われる、非常に単純な生ストリーム用コンテナである。
//! VP9 の仕様書自体にはコンテナフォーマットの定義は含まれないため、ここでは一般に公開されている
//! IVF フォーマットの仕様（32 バイトのファイルヘッダ + フレームごとの 12 バイトヘッダ）に基づいて
//! パーサを実装する。すべての多バイト整数値はリトルエンディアンである。
//!
//! ファイルヘッダのレイアウト（32 バイト）:
//!
//! | オフセット | サイズ | 内容 |
//! | --- | --- | --- |
//! | 0  | 4 | シグネチャ `"DKIF"` |
//! | 4  | 2 | バージョン（0 であるべき） |
//! | 6  | 2 | ヘッダ長（バイト、通常 32） |
//! | 8  | 4 | コーデック FourCC（VP9 の場合 `"VP90"`） |
//! | 12 | 2 | 幅（ピクセル） |
//! | 14 | 2 | 高さ（ピクセル） |
//! | 16 | 4 | タイムベース分母 |
//! | 20 | 4 | タイムベース分子 |
//! | 24 | 4 | フレーム数 |
//! | 28 | 4 | 未使用 |
//!
//! フレームヘッダのレイアウト（12 バイト、フレームデータが後続する）:
//!
//! | オフセット | サイズ | 内容 |
//! | --- | --- | --- |
//! | 0 | 4 | フレームデータのサイズ（このヘッダを含まない） |
//! | 4 | 8 | 64 ビットのプレゼンテーションタイムスタンプ |

/// IVF ファイルヘッダのサイズ（バイト）。
const IVF_FILE_HEADER_SIZE: usize = 32;
/// IVF フレームヘッダのサイズ（バイト）。
const IVF_FRAME_HEADER_SIZE: usize = 12;
/// IVF ファイルの先頭シグネチャ。
const IVF_SIGNATURE: &[u8; 4] = b"DKIF";

/// IVF パース時に発生し得るエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvfError {
    /// バッファがファイルヘッダより短い。
    TooShortForFileHeader,
    /// 先頭 4 バイトが `"DKIF"` ではない。
    BadSignature,
    /// ヘッダに記載された header_length が実際のバッファサイズと矛盾する等、不正な値。
    InvalidHeaderLength,
    /// フレームヘッダを読むためのバイト数が足りない。
    TruncatedFrameHeader,
    /// フレームヘッダが示すデータサイズ分のバイトがバッファに存在しない。
    TruncatedFrameData,
}

/// IVF ファイルヘッダの内容。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfHeader {
    /// フォーマットバージョン（通常 0）。
    pub version: u16,
    /// ヘッダ長（バイト）。通常は 32。
    pub header_length: u16,
    /// コーデック FourCC（例: `[b'V', b'P', b'9', b'0']`）。
    pub fourcc: [u8; 4],
    /// フレーム幅（ピクセル）。
    pub width: u16,
    /// フレーム高さ（ピクセル）。
    pub height: u16,
    /// タイムベースの分母。
    pub timebase_denominator: u32,
    /// タイムベースの分子。
    pub timebase_numerator: u32,
    /// ファイルに含まれるフレーム数（エンコーダの自己申告値であり、実際の数と一致しない場合もある）。
    pub frame_count: u32,
}

/// 1 フレーム分の IVF データ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfFrame<'a> {
    /// 64 ビットのプレゼンテーションタイムスタンプ（timebase 単位）。
    pub timestamp: u64,
    /// フレームの生データ（VP9 の場合、フレームまたはスーパーフレームのバイト列）。
    pub data: &'a [u8],
}

/// IVF ファイルを順にフレームへ分解するリーダー。
///
/// バッファ全体をあらかじめメモリに読み込んでおき、そのスライスを借用する設計とする。
/// `Iterator` を実装しており、`next()` を呼ぶたびに次のフレームを返す。
#[derive(Debug, Clone)]
pub struct IvfReader<'a> {
    header: IvfHeader,
    /// まだ読んでいない残りのバイト列（フレームヘッダ+データの繰り返し）。
    remaining: &'a [u8],
}

impl<'a> IvfReader<'a> {
    /// バッファの先頭から IVF ファイルヘッダを読み取り、`IvfReader` を構築する。
    pub fn new(buf: &'a [u8]) -> Result<Self, IvfError> {
        if buf.len() < IVF_FILE_HEADER_SIZE {
            return Err(IvfError::TooShortForFileHeader);
        }
        if &buf[0..4] != IVF_SIGNATURE {
            return Err(IvfError::BadSignature);
        }
        let version = read_u16_le(buf, 4);
        let header_length = read_u16_le(buf, 6);
        let fourcc = [buf[8], buf[9], buf[10], buf[11]];
        let width = read_u16_le(buf, 12);
        let height = read_u16_le(buf, 14);
        let timebase_denominator = read_u32_le(buf, 16);
        let timebase_numerator = read_u32_le(buf, 20);
        let frame_count = read_u32_le(buf, 24);

        // header_length はファイルヘッダの実サイズを示す。32 バイト未満だと後続データの
        // 開始位置が不定になってしまうため不正値として扱う。
        let header_length_usize = header_length as usize;
        if header_length_usize < IVF_FILE_HEADER_SIZE || header_length_usize > buf.len() {
            return Err(IvfError::InvalidHeaderLength);
        }

        Ok(Self {
            header: IvfHeader {
                version,
                header_length,
                fourcc,
                width,
                height,
                timebase_denominator,
                timebase_numerator,
                frame_count,
            },
            remaining: &buf[header_length_usize..],
        })
    }

    /// パース済みの IVF ファイルヘッダを返す。
    pub fn header(&self) -> &IvfHeader {
        &self.header
    }
}

impl<'a> Iterator for IvfReader<'a> {
    type Item = Result<IvfFrame<'a>, IvfError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            return None;
        }
        if self.remaining.len() < IVF_FRAME_HEADER_SIZE {
            // これ以上フレームヘッダを読めない半端なデータが残っている。
            self.remaining = &[];
            return Some(Err(IvfError::TruncatedFrameHeader));
        }

        let frame_size = read_u32_le(self.remaining, 0) as usize;
        let timestamp = read_u64_le(self.remaining, 4);

        let data_start = IVF_FRAME_HEADER_SIZE;
        let data_end = data_start + frame_size;
        if self.remaining.len() < data_end {
            self.remaining = &[];
            return Some(Err(IvfError::TruncatedFrameData));
        }

        let data = &self.remaining[data_start..data_end];
        self.remaining = &self.remaining[data_end..];

        Some(Ok(IvfFrame { timestamp, data }))
    }
}

fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用に、指定したフィールドから IVF ファイルヘッダのバイト列を手組みする。
    fn build_file_header(
        fourcc: &[u8; 4],
        width: u16,
        height: u16,
        timebase_den: u32,
        timebase_num: u32,
        frame_count: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(IVF_FILE_HEADER_SIZE);
        buf.extend_from_slice(b"DKIF");
        buf.extend_from_slice(&0u16.to_le_bytes()); // version
        buf.extend_from_slice(&32u16.to_le_bytes()); // header_length
        buf.extend_from_slice(fourcc);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&timebase_den.to_le_bytes());
        buf.extend_from_slice(&timebase_num.to_le_bytes());
        buf.extend_from_slice(&frame_count.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // unused
        assert_eq!(buf.len(), IVF_FILE_HEADER_SIZE);
        buf
    }

    /// テスト用に、1 フレーム分の 12 バイトヘッダ + データを手組みする。
    fn append_frame(buf: &mut Vec<u8>, timestamp: u64, data: &[u8]) {
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&timestamp.to_le_bytes());
        buf.extend_from_slice(data);
    }

    #[test]
    fn parses_file_header_fields() {
        let mut buf = build_file_header(b"VP90", 352, 288, 30, 1, 2);
        append_frame(&mut buf, 0, &[0xAA, 0xBB, 0xCC]);
        append_frame(&mut buf, 1, &[0x11, 0x22]);

        let reader = IvfReader::new(&buf).expect("valid header");
        let header = reader.header();
        assert_eq!(header.version, 0);
        assert_eq!(header.header_length, 32);
        assert_eq!(&header.fourcc, b"VP90");
        assert_eq!(header.width, 352);
        assert_eq!(header.height, 288);
        assert_eq!(header.timebase_denominator, 30);
        assert_eq!(header.timebase_numerator, 1);
        assert_eq!(header.frame_count, 2);
    }

    #[test]
    fn iterates_frames_in_order() {
        let mut buf = build_file_header(b"VP90", 16, 16, 30, 1, 3);
        append_frame(&mut buf, 0, &[1, 2, 3]);
        append_frame(&mut buf, 33, &[4, 5]);
        append_frame(&mut buf, 66, &[]);

        let reader = IvfReader::new(&buf).expect("valid header");
        let frames: Vec<IvfFrame> = reader.map(|f| f.expect("frame ok")).collect();

        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].timestamp, 0);
        assert_eq!(frames[0].data, &[1, 2, 3]);
        assert_eq!(frames[1].timestamp, 33);
        assert_eq!(frames[1].data, &[4, 5]);
        assert_eq!(frames[2].timestamp, 66);
        assert_eq!(frames[2].data, &[] as &[u8]);
    }

    #[test]
    fn rejects_bad_signature() {
        let mut buf = build_file_header(b"VP90", 16, 16, 30, 1, 1);
        buf[0] = b'X'; // シグネチャを壊す
        assert_eq!(IvfReader::new(&buf).unwrap_err(), IvfError::BadSignature);
    }

    #[test]
    fn rejects_too_short_buffer() {
        let buf = vec![0u8; 10];
        assert_eq!(
            IvfReader::new(&buf).unwrap_err(),
            IvfError::TooShortForFileHeader
        );
    }

    #[test]
    fn reports_truncated_frame_data() {
        let mut buf = build_file_header(b"VP90", 16, 16, 30, 1, 1);
        // フレームサイズを 10 と主張するが、実際には 2 バイトしか続かない不正なデータ。
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&[0xAA, 0xBB]);

        let mut reader = IvfReader::new(&buf).expect("valid header");
        assert_eq!(reader.next(), Some(Err(IvfError::TruncatedFrameData)));
    }

    #[test]
    fn handles_empty_stream() {
        let buf = build_file_header(b"VP90", 16, 16, 30, 1, 0);
        let mut reader = IvfReader::new(&buf).expect("valid header");
        assert_eq!(reader.next(), None);
    }
}
