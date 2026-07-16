//! IVF container parser.
//!
//! IVF is a very simple raw-stream container commonly used for libvpx (VP8/VP9)
//! testing and distribution. The VP9 spec itself does not define a container
//! format, so this parser is implemented based on the publicly documented IVF
//! format spec (a 32-byte file header + a 12-byte header per frame). All
//! multi-byte integer values are little-endian.
//!
//! File header layout (32 bytes):
//!
//! | Offset | Size | Contents |
//! | --- | --- | --- |
//! | 0  | 4 | Signature `"DKIF"` |
//! | 4  | 2 | Version (should be 0) |
//! | 6  | 2 | Header length (bytes, usually 32) |
//! | 8  | 4 | Codec FourCC (`"VP90"` for VP9) |
//! | 12 | 2 | Width (pixels) |
//! | 14 | 2 | Height (pixels) |
//! | 16 | 4 | Timebase denominator |
//! | 20 | 4 | Timebase numerator |
//! | 24 | 4 | Frame count |
//! | 28 | 4 | Unused |
//!
//! Frame header layout (12 bytes, followed by frame data):
//!
//! | Offset | Size | Contents |
//! | --- | --- | --- |
//! | 0 | 4 | Size of the frame data (excluding this header) |
//! | 4 | 8 | 64-bit presentation timestamp |

/// Size of the IVF file header (bytes).
const IVF_FILE_HEADER_SIZE: usize = 32;
/// Size of the IVF frame header (bytes).
const IVF_FRAME_HEADER_SIZE: usize = 12;
/// The signature at the start of an IVF file.
const IVF_SIGNATURE: &[u8; 4] = b"DKIF";

/// Errors that can occur while parsing IVF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvfError {
    /// The buffer is shorter than the file header.
    TooShortForFileHeader,
    /// The first 4 bytes are not `"DKIF"`.
    BadSignature,
    /// An invalid value, e.g. the header_length in the header is inconsistent with the actual buffer size.
    InvalidHeaderLength,
    /// Not enough bytes remain to read a frame header.
    TruncatedFrameHeader,
    /// The buffer does not contain as many bytes as the frame header claims for the data size.
    TruncatedFrameData,
}

/// Contents of the IVF file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfHeader {
    /// Format version (usually 0).
    pub version: u16,
    /// Header length (bytes). Usually 32.
    pub header_length: u16,
    /// Codec FourCC (e.g. `[b'V', b'P', b'9', b'0']`).
    pub fourcc: [u8; 4],
    /// Frame width (pixels).
    pub width: u16,
    /// Frame height (pixels).
    pub height: u16,
    /// Timebase denominator.
    pub timebase_denominator: u32,
    /// Timebase numerator.
    pub timebase_numerator: u32,
    /// Number of frames contained in the file (self-reported by the encoder; may not match the actual count).
    pub frame_count: u32,
}

/// IVF data for a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfFrame<'a> {
    /// 64-bit presentation timestamp (in timebase units).
    pub timestamp: u64,
    /// Raw frame data (for VP9, the byte sequence of a frame or superframe).
    pub data: &'a [u8],
}

/// A reader that splits an IVF file into frames in order.
///
/// Designed around loading the entire buffer into memory up front and
/// borrowing a slice of it. Implements `Iterator`, returning the next frame
/// on each call to `next()`.
#[derive(Debug, Clone)]
pub struct IvfReader<'a> {
    header: IvfHeader,
    /// The remaining unread byte sequence (repeated frame header + data).
    remaining: &'a [u8],
}

impl<'a> IvfReader<'a> {
    /// Reads the IVF file header from the start of the buffer and constructs an `IvfReader`.
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

        // header_length indicates the actual size of the file header. If it is
        // less than 32 bytes, the start position of the following data would
        // be indeterminate, so treat it as an invalid value.
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

    /// Returns the parsed IVF file header.
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
            // What remains is a partial chunk too short to read another frame header from.
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

/// Builds a complete IVF file from `frames`, the inverse of [`IvfReader`] (matches its layout
/// field-for-field). Each frame's timestamp is simply its index. Useful for tests that need to
/// hand-build an IVF file, and for M4 vector tooling.
pub fn write_ivf(
    fourcc: &[u8; 4],
    width: u16,
    height: u16,
    timebase_den: u32,
    timebase_num: u32,
    frames: &[Vec<u8>],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(IVF_SIGNATURE);
    buf.extend_from_slice(&0u16.to_le_bytes()); // version
    buf.extend_from_slice(&(IVF_FILE_HEADER_SIZE as u16).to_le_bytes());
    buf.extend_from_slice(fourcc);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(&timebase_den.to_le_bytes());
    buf.extend_from_slice(&timebase_num.to_le_bytes());
    buf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // unused

    for (i, frame) in frames.iter().enumerate() {
        buf.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(i as u64).to_le_bytes());
        buf.extend_from_slice(frame);
    }
    buf
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

    /// Hand-builds the byte sequence of an IVF file header from the given fields, for tests
    /// that need a `frame_count` or per-frame timestamps independent of the actual frames
    /// appended (unlike `write_ivf`, which always derives both from `frames`).
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

    /// Hand-builds the 12-byte header + data for a single frame, for tests.
    fn append_frame(buf: &mut Vec<u8>, timestamp: u64, data: &[u8]) {
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&timestamp.to_le_bytes());
        buf.extend_from_slice(data);
    }

    #[test]
    fn parses_file_header_fields() {
        let buf = write_ivf(
            b"VP90",
            352,
            288,
            30,
            1,
            &[vec![0xAA, 0xBB, 0xCC], vec![0x11, 0x22]],
        );

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
        let mut buf = write_ivf(b"VP90", 16, 16, 30, 1, &[]);
        buf[0] = b'X'; // Corrupt the signature
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
        // Claims a frame size of 10, but only 2 bytes of data actually follow; invalid.
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&[0xAA, 0xBB]);

        let mut reader = IvfReader::new(&buf).expect("valid header");
        assert_eq!(reader.next(), Some(Err(IvfError::TruncatedFrameData)));
    }

    #[test]
    fn handles_empty_stream() {
        let buf = write_ivf(b"VP90", 16, 16, 30, 1, &[]);
        let mut reader = IvfReader::new(&buf).expect("valid header");
        assert_eq!(reader.next(), None);
    }
}
