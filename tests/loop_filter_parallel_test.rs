//! The intra-plane wavefront loop filter must be byte-identical to the sequential loop filter.
//!
//! HD clips (frame size >= `LF_PARALLEL_MIN_MI`) take the wavefront path
//! (`loop_filter::wavefront_filter_plane`: several worker threads per plane, each superblock row
//! lagging the one above by two superblocks). This test decodes each clip with the wavefront
//! engaged and again with the test-only `FORCE_SEQUENTIAL_LOOP_FILTER` knob, and requires every
//! displayed frame to match exactly -- pinning the `unsafe` shared-buffer wavefront against the
//! safe sequential reference. The parallel decode is repeated several times to give any
//! timing-dependent race a chance to surface. Skips cleanly if the vectors haven't been downloaded.

mod common;

use std::sync::atomic::Ordering;

use vp9dec::ivf::IvfReader;
use vp9dec::loop_filter::FORCE_SEQUENTIAL_LOOP_FILTER;
use vp9dec::Decoder;

/// Clips large enough to engage the loop-filter wavefront, kept small in file size so decoding
/// each one several times stays fast. 1080p and 720p both clear `LF_PARALLEL_MIN_MI`.
const HD_VECTORS: [&str; 2] = [
    "vp90-2-02-size-lf-1920x1080.ivf",
    "vp90-2-22-svc_1280x720_1.ivf",
];

fn decode_displayed_frames(ivf_bytes: &[u8]) -> Vec<Vec<u8>> {
    let reader = IvfReader::new(ivf_bytes).expect("failed to parse IVF header");
    let mut decoder = Decoder::new();
    let mut out = Vec::new();
    for frame in reader {
        let frame = frame.expect("failed to read IVF frame");
        let decoded = decoder.decode_frame(frame.data).expect("decode failed");
        for d in decoded {
            if let Some(f) = d.frame {
                out.push(common::i420_bytes(&f));
            }
        }
    }
    out
}

#[test]
fn wavefront_loop_filter_matches_sequential_byte_for_byte() {
    // Repeat the parallel decode so a timing-dependent race has several chances to diverge from
    // the deterministic sequential reference.
    const PARALLEL_ITERS: usize = 5;

    for name in HD_VECTORS {
        let Some(bytes) = common::read_vector(name) else {
            continue;
        };

        // Reference: the safe sequential loop filter.
        FORCE_SEQUENTIAL_LOOP_FILTER.store(true, Ordering::Relaxed);
        let sequential = decode_displayed_frames(&bytes);
        FORCE_SEQUENTIAL_LOOP_FILTER.store(false, Ordering::Relaxed);

        assert!(
            !sequential.is_empty(),
            "{name}: no displayed frames decoded (bad vector?)"
        );

        for iter in 0..PARALLEL_ITERS {
            let parallel = decode_displayed_frames(&bytes);
            assert_eq!(
                parallel.len(),
                sequential.len(),
                "{name}: displayed frame count differs (iter {iter})"
            );
            for (i, (p, s)) in parallel.iter().zip(sequential.iter()).enumerate() {
                assert_eq!(
                    p, s,
                    "{name}: frame {i} differs, wavefront vs sequential (iter {iter})"
                );
            }
        }
        eprintln!(
            "{name}: {} displayed frames byte-identical, wavefront vs sequential ({PARALLEL_ITERS} parallel iters)",
            sequential.len()
        );
    }
}
