//! Remuxes a WebM file's VP9 video track into a bare IVF elementary stream.
//!
//! Container change only -- no re-encode. Each WebM (Simple)Block's frame payload (the
//! VP9 elementary-stream bytes, possibly a superframe) is copied verbatim into an IVF
//! frame record. The whole thing is std-only: a small EBML/Matroska reader, just enough
//! to walk these conformance vectors, lives in this file (the decoder crate itself, and
//! this repo as a whole, has zero dependencies -- see README).
//!
//! IVF layout produced (matches `src/ivf.rs`'s `IvfReader`, all little-endian):
//! File header (32 bytes): "DKIF" + version(u16) + header_length(u16) + fourcc(4) +
//! width(u16) + height(u16) + timebase_denominator(u32) + timebase_numerator(u32) +
//! frame_count(u32) + unused(u32).
//! Per-frame record: frame_size(u32) + timestamp(u64) + frame_size bytes of data.
//!
//! Usage: `cargo run --example webm_to_ivf -- <in.webm> <out.ivf>`

use std::process::exit;

// --- EBML element IDs (retain their length-marker bits, per the EBML spec) ---
const ID_EBML_HEADER: u64 = 0x1A45_DFA3;
const ID_SEGMENT: u64 = 0x1853_8067;
const ID_TRACKS: u64 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u64 = 0xAE;
const ID_TRACK_NUMBER: u64 = 0xD7;
const ID_TRACK_TYPE: u64 = 0x83;
const ID_CODEC_ID: u64 = 0x86;
const ID_VIDEO: u64 = 0xE0;
const ID_PIXEL_WIDTH: u64 = 0xB0;
const ID_PIXEL_HEIGHT: u64 = 0xBA;
const ID_CLUSTER: u64 = 0x1F43_B675;
const ID_SIMPLE_BLOCK: u64 = 0xA3;
const ID_BLOCK_GROUP: u64 = 0xA0;
const ID_BLOCK: u64 = 0xA1;

// Other Segment-level (level-1) IDs. Needed only to know where an unknown-sized Cluster
// ends: an unknown-sized master runs until the next element that belongs to an equal or
// higher level, which at Segment scope means any of these.
const SEGMENT_LEVEL_IDS: &[u64] = &[
    ID_SEGMENT,
    ID_TRACKS,
    ID_CLUSTER,
    0x114D_9B74, // SeekHead
    0x1549_A966, // Info
    0x1C53_BB6B, // Cues
    0x1043_A770, // Chapters
    0x1254_C367, // Tags
    0x1941_A469, // Attachments
];

const TRACK_TYPE_VIDEO: u64 = 1;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (in_path, out_path) = match args.as_slice() {
        [in_path, out_path] => (in_path.clone(), out_path.clone()),
        _ => {
            eprintln!("usage: webm_to_ivf <in.webm> <out.ivf>");
            exit(1);
        }
    };

    let buf =
        std::fs::read(&in_path).unwrap_or_else(|e| fail(format!("failed to read {in_path}: {e}")));

    let segment = find_segment(&buf, &in_path);
    let track = find_vp9_track(&buf, segment.clone(), &in_path);
    let payloads = collect_frames(&buf, segment, track.number, &in_path);

    write_ivf(&out_path, track.width, track.height, &payloads, &in_path);
    eprintln!(
        "[ok] {in_path} -> {out_path} ({}x{}, {} frames)",
        track.width,
        track.height,
        payloads.len()
    );
}

fn fail(msg: String) -> ! {
    eprintln!("[error] {msg}");
    exit(1);
}

/// A parsed element header: its ID and the byte range of its *content* (`start..end`).
/// For an unknown-sized element, `end` is left at the enclosing bound and the caller
/// resolves the true end structurally.
#[derive(Clone)]
struct Elem {
    id: u64,
    start: usize,
    end: usize,
    size_known: bool,
}

/// Reads an element header (ID vint + size vint) starting at `pos`, bounded by `outer_end`.
fn read_header(buf: &[u8], pos: usize, outer_end: usize, ctx: &str) -> Elem {
    let (id, after_id) = read_id(buf, pos, outer_end, ctx);
    let (size_opt, after_size) = read_size(buf, after_id, outer_end, ctx);
    match size_opt {
        Some(size) => {
            let end = after_size
                .checked_add(size as usize)
                .filter(|&e| e <= outer_end)
                .unwrap_or_else(|| {
                    fail(format!(
                        "{ctx}: element 0x{id:X} size {size} overruns its container"
                    ))
                });
            Elem {
                id,
                start: after_size,
                end,
                size_known: true,
            }
        }
        None => Elem {
            id,
            start: after_size,
            end: outer_end,
            size_known: false,
        },
    }
}

/// Reads an EBML element ID vint, keeping the length-marker bits (IDs are stored verbatim).
fn read_id(buf: &[u8], pos: usize, end: usize, ctx: &str) -> (u64, usize) {
    let (first, length) = vint_len(buf, pos, end, ctx);
    let _ = first;
    let mut value = 0u64;
    for &b in &buf[pos..pos + length] {
        value = (value << 8) | b as u64;
    }
    (value, pos + length)
}

/// Reads an EBML size vint, returning `None` for the all-ones "unknown size".
fn read_size(buf: &[u8], pos: usize, end: usize, ctx: &str) -> (Option<u64>, usize) {
    let (first, length) = vint_len(buf, pos, end, ctx);
    let mask = first_byte_mask(length);
    let mut value = (first & mask) as u64;
    let mut all_ones = (first & mask) == mask;
    for &b in &buf[pos + 1..pos + length] {
        value = (value << 8) | b as u64;
        all_ones &= b == 0xFF;
    }
    if all_ones {
        (None, pos + length)
    } else {
        (Some(value), pos + length)
    }
}

/// Reads a plain EBML vint value with its marker bit stripped (track numbers, etc.).
fn read_vint_value(buf: &[u8], pos: usize, end: usize, ctx: &str) -> (u64, usize) {
    let (first, length) = vint_len(buf, pos, end, ctx);
    let mask = first_byte_mask(length);
    let mut value = (first & mask) as u64;
    for &b in &buf[pos + 1..pos + length] {
        value = (value << 8) | b as u64;
    }
    (value, pos + length)
}

/// Mask isolating the value bits of a vint's first byte (the marker bit and any higher
/// length-descriptor bits cleared). For an 8-byte vint the first byte is pure descriptor,
/// so the mask is 0 (`0xFF >> 8` would overflow a u8).
fn first_byte_mask(length: usize) -> u8 {
    if length >= 8 {
        0
    } else {
        0xFFu8 >> length
    }
}

/// Determines a vint's length from its leading byte and validates it fits in `[pos, end)`.
fn vint_len(buf: &[u8], pos: usize, end: usize, ctx: &str) -> (u8, usize) {
    if pos >= end {
        fail(format!(
            "{ctx}: unexpected end of data while reading a vint"
        ));
    }
    let first = buf[pos];
    if first == 0 {
        fail(format!(
            "{ctx}: invalid vint (leading byte 0x00) at offset {pos}"
        ));
    }
    let length = first.leading_zeros() as usize + 1; // 0x80 -> 1, 0x40 -> 2, ...
    if pos + length > end {
        fail(format!(
            "{ctx}: vint of length {length} runs past the container at offset {pos}"
        ));
    }
    (first, length)
}

/// Locates the single Segment element and returns its content range.
fn find_segment(buf: &[u8], ctx: &str) -> std::ops::Range<usize> {
    let mut pos = 0usize;
    let file_end = buf.len();
    while pos < file_end {
        let elem = read_header(buf, pos, file_end, ctx);
        if elem.id == ID_SEGMENT {
            return elem.start..elem.end;
        }
        if !elem.size_known {
            fail(format!(
                "{ctx}: unexpected unknown-sized top-level element 0x{:X}",
                elem.id
            ));
        }
        if elem.id != ID_EBML_HEADER {
            eprintln!(
                "[note] {ctx}: skipping top-level element 0x{:X} before Segment",
                elem.id
            );
        }
        pos = elem.end;
    }
    fail(format!("{ctx}: no Segment element found"));
}

struct Vp9Track {
    number: u64,
    width: u16,
    height: u16,
}

/// Scans Segment children for the Tracks element, then finds the single V_VP9 video track.
fn find_vp9_track(buf: &[u8], segment: std::ops::Range<usize>, ctx: &str) -> Vp9Track {
    let mut pos = segment.start;
    while pos < segment.end {
        let elem = read_header(buf, pos, segment.end, ctx);
        if elem.id == ID_TRACKS {
            return parse_tracks(buf, elem.start..elem.end, ctx);
        }
        if !elem.size_known {
            // Only Cluster is expected unknown-sized; Tracks always precedes the clusters
            // in these files, so an unknown size before Tracks means we mis-parsed.
            fail(format!(
                "{ctx}: reached unknown-sized element 0x{:X} before Tracks",
                elem.id
            ));
        }
        pos = elem.end;
    }
    fail(format!("{ctx}: no Tracks element found in Segment"));
}

fn parse_tracks(buf: &[u8], tracks: std::ops::Range<usize>, ctx: &str) -> Vp9Track {
    let mut pos = tracks.start;
    while pos < tracks.end {
        let elem = read_header(buf, pos, tracks.end, ctx);
        if elem.id == ID_TRACK_ENTRY {
            if let Some(track) = parse_track_entry(buf, elem.start..elem.end, ctx) {
                return track;
            }
        }
        pos = elem.end;
    }
    fail(format!("{ctx}: no V_VP9 video track found in Tracks"));
}

/// Parses one TrackEntry; returns `Some` only if it is a V_VP9 video track.
fn parse_track_entry(buf: &[u8], entry: std::ops::Range<usize>, ctx: &str) -> Option<Vp9Track> {
    let mut number: Option<u64> = None;
    let mut track_type: Option<u64> = None;
    let mut codec_id: Option<String> = None;
    let mut dims: Option<(u16, u16)> = None;

    let mut pos = entry.start;
    while pos < entry.end {
        let elem = read_header(buf, pos, entry.end, ctx);
        match elem.id {
            ID_TRACK_NUMBER => number = Some(read_uint(buf, elem.start..elem.end)),
            ID_TRACK_TYPE => track_type = Some(read_uint(buf, elem.start..elem.end)),
            ID_CODEC_ID => codec_id = Some(read_string(buf, elem.start..elem.end)),
            ID_VIDEO => dims = Some(parse_video(buf, elem.start..elem.end, ctx)),
            _ => {}
        }
        pos = elem.end;
    }

    if track_type == Some(TRACK_TYPE_VIDEO) && codec_id.as_deref() == Some("V_VP9") {
        let number =
            number.unwrap_or_else(|| fail(format!("{ctx}: V_VP9 track has no TrackNumber")));
        let (width, height) = dims
            .unwrap_or_else(|| fail(format!("{ctx}: V_VP9 track has no Video/pixel dimensions")));
        Some(Vp9Track {
            number,
            width,
            height,
        })
    } else {
        None
    }
}

fn parse_video(buf: &[u8], video: std::ops::Range<usize>, ctx: &str) -> (u16, u16) {
    let mut width: Option<u64> = None;
    let mut height: Option<u64> = None;
    let mut pos = video.start;
    while pos < video.end {
        let elem = read_header(buf, pos, video.end, ctx);
        match elem.id {
            ID_PIXEL_WIDTH => width = Some(read_uint(buf, elem.start..elem.end)),
            ID_PIXEL_HEIGHT => height = Some(read_uint(buf, elem.start..elem.end)),
            _ => {}
        }
        pos = elem.end;
    }
    let width = width.unwrap_or_else(|| fail(format!("{ctx}: Video element has no PixelWidth")));
    let height = height.unwrap_or_else(|| fail(format!("{ctx}: Video element has no PixelHeight")));
    let to_u16 = |v: u64, name: &str| {
        u16::try_from(v)
            .unwrap_or_else(|_| fail(format!("{ctx}: {name} {v} exceeds IVF's u16 field")))
    };
    (to_u16(width, "PixelWidth"), to_u16(height, "PixelHeight"))
}

/// Walks all Clusters in file order and extracts every block payload on `track_number`.
fn collect_frames(
    buf: &[u8],
    segment: std::ops::Range<usize>,
    track_number: u64,
    ctx: &str,
) -> Vec<Vec<u8>> {
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut pos = segment.start;
    while pos < segment.end {
        let elem = read_header(buf, pos, segment.end, ctx);
        if elem.id == ID_CLUSTER {
            let cluster_end = if elem.size_known {
                elem.end
            } else {
                find_unknown_master_end(buf, elem.start, segment.end, ctx)
            };
            parse_cluster(
                buf,
                elem.start..cluster_end,
                track_number,
                ctx,
                &mut payloads,
            );
            pos = cluster_end;
        } else if elem.size_known {
            pos = elem.end;
        } else {
            fail(format!(
                "{ctx}: unexpected unknown-sized element 0x{:X} at Segment level",
                elem.id
            ));
        }
    }
    payloads
}

/// For an unknown-sized master at Segment level, scans forward and returns the offset of
/// the next equal-or-higher-level element (or the Segment end), which is where it ends.
fn find_unknown_master_end(
    buf: &[u8],
    content_start: usize,
    segment_end: usize,
    ctx: &str,
) -> usize {
    let mut pos = content_start;
    while pos < segment_end {
        let elem = read_header(buf, pos, segment_end, ctx);
        if SEGMENT_LEVEL_IDS.contains(&elem.id) {
            return pos;
        }
        if !elem.size_known {
            fail(format!(
                "{ctx}: nested unknown-sized element 0x{:X} inside an unknown-sized Cluster",
                elem.id
            ));
        }
        pos = elem.end;
    }
    segment_end
}

fn parse_cluster(
    buf: &[u8],
    cluster: std::ops::Range<usize>,
    track_number: u64,
    ctx: &str,
    payloads: &mut Vec<Vec<u8>>,
) {
    let mut pos = cluster.start;
    while pos < cluster.end {
        let elem = read_header(buf, pos, cluster.end, ctx);
        match elem.id {
            ID_SIMPLE_BLOCK => {
                if let Some(p) = block_payload(buf, elem.start..elem.end, track_number, ctx) {
                    payloads.push(p.to_vec());
                }
            }
            ID_BLOCK_GROUP => {
                // A BlockGroup wraps exactly one Block (plus optional metadata we ignore).
                let mut gpos = elem.start;
                while gpos < elem.end {
                    let inner = read_header(buf, gpos, elem.end, ctx);
                    if inner.id == ID_BLOCK {
                        if let Some(p) =
                            block_payload(buf, inner.start..inner.end, track_number, ctx)
                        {
                            payloads.push(p.to_vec());
                        }
                    }
                    gpos = inner.end;
                }
            }
            _ => {}
        }
        pos = elem.end;
    }
}

/// Parses a (Simple)Block body and returns the frame payload slice iff it belongs to
/// `track_number`. Block body layout: track-number vint, 2-byte signed relative timecode,
/// 1 flags byte, then the payload. Errors out on any lacing (VP9-in-WebM is one frame per
/// block; a VP9 superframe is an opaque payload-internal concept and passes through whole).
fn block_payload<'a>(
    buf: &'a [u8],
    block: std::ops::Range<usize>,
    track_number: u64,
    ctx: &str,
) -> Option<&'a [u8]> {
    let (track, after_track) = read_vint_value(buf, block.start, block.end, ctx);
    if track != track_number {
        return None;
    }
    // 2-byte relative timecode (unused: IVF uses the frame index) + 1 flags byte.
    let flags_pos = after_track + 2;
    if flags_pos >= block.end {
        fail(format!("{ctx}: block body truncated before flags byte"));
    }
    let flags = buf[flags_pos];
    let lacing = (flags >> 1) & 0x03;
    if lacing != 0 {
        fail(format!(
            "{ctx}: encountered a laced block (lacing bits = {lacing}) on the VP9 track -- \
             refusing to guess how to split it, please advise"
        ));
    }
    Some(&buf[flags_pos + 1..block.end])
}

fn read_uint(buf: &[u8], range: std::ops::Range<usize>) -> u64 {
    let mut value = 0u64;
    for &b in &buf[range] {
        value = (value << 8) | b as u64;
    }
    value
}

fn read_string(buf: &[u8], range: std::ops::Range<usize>) -> String {
    String::from_utf8_lossy(&buf[range]).into_owned()
}

fn write_ivf(out_path: &str, width: u16, height: u16, payloads: &[Vec<u8>], ctx: &str) {
    // timebase 1/1 (unused by our IvfReader) is this example's long-standing output format;
    // kept as-is rather than switched to a real frame rate, to not change existing output.
    let out = vp9dec::ivf::write_ivf(b"VP90", width, height, 1, 1, payloads);
    std::fs::write(out_path, &out)
        .unwrap_or_else(|e| fail(format!("{ctx}: failed to write {out_path}: {e}")));
}
