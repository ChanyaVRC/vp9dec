//! VP9 "superframe" splitting (spec "VP9 Bitstream - superframe and uncompressed header" §3).
//!
//! A single container chunk (one IVF frame, one WebM block, etc.) may pack more than one
//! VP9 frame -- commonly one or more hidden frames (`show_frame == 0`, e.g. an altref, or a
//! sequence of intra-only frames priming several DPB slots) followed by one visible frame.
//! This is signaled by a trailing "superframe index": if the chunk's last byte has its top 3
//! bits set to `0b110`, the next 2 bits give `frame_size_length_minus_one` and the low 3 bits
//! give `num_frames_minus_one`; the index (of size `2 + (frame_size_length_minus_one + 1) *
//! (num_frames_minus_one + 1)` bytes, repeating the marker byte at both ends) then lists each
//! frame's byte size (least-significant byte first), and the chunk's payload up to the index
//! is exactly the concatenation of that many consecutive VP9 frames, in decode order.
//!
//! A container reader (e.g. [`crate::ivf::IvfReader`]) hands back one chunk at a time; callers
//! must run each chunk through [`split_superframe`] before feeding the result to
//! [`crate::Decoder::decode_frame`], which itself decodes exactly one VP9 frame.

/// Splits one container chunk into the VP9 frame(s) it contains, in decode order.
///
/// If `data` doesn't end with a valid superframe index, returns `data` itself as the sole
/// element (the common case: one chunk == one VP9 frame).
pub fn split_superframe(data: &[u8]) -> Vec<&[u8]> {
    parse_superframe_index(data).unwrap_or_else(|| vec![data])
}

fn parse_superframe_index(data: &[u8]) -> Option<Vec<&[u8]>> {
    let marker = *data.last()?;
    if marker & 0xe0 != 0xc0 {
        return None;
    }
    let bytes_per_framesize = (((marker >> 3) & 0x3) + 1) as usize;
    let num_frames = ((marker & 0x7) + 1) as usize;
    let index_size = 2 + bytes_per_framesize * num_frames;
    if data.len() < index_size {
        return None;
    }
    let index = &data[data.len() - index_size..];
    // The marker byte is duplicated at the start of the index; a mismatch means this isn't
    // actually a superframe index (spec: "go to the first byte of the superframe index, and
    // check that it matches the last byte of the superframe index").
    if index[0] != marker {
        return None;
    }

    let mut sizes = Vec::with_capacity(num_frames);
    let mut pos = 1;
    for _ in 0..num_frames {
        let mut size = 0usize;
        for (b, &byte) in index[pos..pos + bytes_per_framesize].iter().enumerate() {
            size |= (byte as usize) << (8 * b);
        }
        sizes.push(size);
        pos += bytes_per_framesize;
    }

    let payload = &data[..data.len() - index_size];
    let mut frames = Vec::with_capacity(num_frames);
    let mut offset = 0usize;
    for size in sizes {
        let end = offset.checked_add(size)?;
        frames.push(payload.get(offset..end)?);
        offset = end;
    }
    Some(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-builds a superframe index for the given frame sizes, using 1-byte frame sizes.
    fn build_index_1byte(frame_sizes: &[u8]) -> Vec<u8> {
        let marker = 0xc0 | ((0u8) << 3) | (frame_sizes.len() as u8 - 1);
        let mut index = vec![marker];
        index.extend_from_slice(frame_sizes);
        index.push(marker);
        index
    }

    #[test]
    fn passes_through_data_with_no_superframe_marker() {
        let data = [1u8, 2, 3, 4, 5];
        assert_eq!(split_superframe(&data), vec![&data[..]]);
    }

    #[test]
    fn splits_a_two_frame_superframe() {
        let mut data = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        // frame 0 = data[0..3] (size 3), frame 1 = data[3..5] (size 2)
        data.extend(build_index_1byte(&[3, 2]));

        let frames = split_superframe(&data);
        assert_eq!(frames, vec![&[0xAA, 0xBB, 0xCC][..], &[0xDD, 0xEE][..]]);
    }

    #[test]
    fn splits_a_four_frame_superframe_matching_the_intra_only_vector_layout() {
        let f0 = vec![1u8; 3];
        let f1 = vec![2u8; 5];
        let f2 = vec![3u8; 7];
        let f3 = vec![4u8; 2];
        let mut data = Vec::new();
        data.extend_from_slice(&f0);
        data.extend_from_slice(&f1);
        data.extend_from_slice(&f2);
        data.extend_from_slice(&f3);
        data.extend(build_index_1byte(&[3, 5, 7, 2]));

        let frames = split_superframe(&data);
        assert_eq!(frames, vec![&f0[..], &f1[..], &f2[..], &f3[..]]);
    }

    #[test]
    fn rejects_index_whose_leading_and_trailing_marker_bytes_disagree() {
        let mut data = vec![0xAAu8, 0xBB, 0xCC];
        // Trailing marker (0xc0: frame_size_length=1, num_frames=1) puts the index start
        // right before the leading byte below, which deliberately doesn't repeat 0xc0.
        data.push(0x00);
        data.push(3);
        data.push(0xc0);
        assert_eq!(split_superframe(&data), vec![&data[..]]);
    }

    #[test]
    fn rejects_index_whose_declared_sizes_overflow_the_payload() {
        let mut data = vec![0xAAu8, 0xBB, 0xCC];
        data.extend(build_index_1byte(&[100])); // claims a 100-byte frame; payload is only 3 bytes
        assert_eq!(split_superframe(&data), vec![&data[..]]);
    }

    #[test]
    fn treats_too_short_a_buffer_as_a_single_frame() {
        // Last byte looks like a marker, but the buffer is far too short for the claimed index.
        let data = [0xffu8];
        assert_eq!(split_superframe(&data), vec![&data[..]]);
    }
}
