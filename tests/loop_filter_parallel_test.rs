//! The intra-plane wavefront loop filter must be byte-identical to the sequential loop filter.
//!
//! HD clips (frame size >= `LF_PARALLEL_MIN_MI`) take the wavefront path
//! (`loop_filter::wavefront_filter_planes`: several worker threads per plane, each superblock row
//! lagging the one above by two superblocks). This test decodes each clip with the wavefront
//! engaged and again with a safe sequential override, and requires a bounded prefix of displayed
//! frames to match exactly. The parallel decode is repeated several times to give any
//! timing-dependent race a chance to surface without retaining gigabytes of frame data. Skips
//! cleanly if the vectors haven't been downloaded.

mod common;

use std::sync::atomic::Ordering;

use vp9dec::ivf::IvfReader;
use vp9dec::loop_filter::{
    FORCE_PARALLEL_LOOP_FILTER, FORCE_SEQUENTIAL_LOOP_FILTER, WAVEFRONT_INVOCATIONS,
};
use vp9dec::Decoder;

const DISPLAYED_FRAMES_PER_VECTOR: usize = 8;

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
                if out.len() == DISPLAYED_FRAMES_PER_VECTOR {
                    return out;
                }
            }
        }
    }
    out
}

/// Restores both process-global test overrides even if a decode or assertion panics.
struct LoopFilterOverride;

impl LoopFilterOverride {
    fn sequential() -> Self {
        FORCE_PARALLEL_LOOP_FILTER.store(false, Ordering::Relaxed);
        FORCE_SEQUENTIAL_LOOP_FILTER.store(true, Ordering::Relaxed);
        Self
    }

    fn parallel() -> Self {
        FORCE_SEQUENTIAL_LOOP_FILTER.store(false, Ordering::Relaxed);
        FORCE_PARALLEL_LOOP_FILTER.store(true, Ordering::Relaxed);
        Self
    }
}

impl Drop for LoopFilterOverride {
    fn drop(&mut self) {
        FORCE_SEQUENTIAL_LOOP_FILTER.store(false, Ordering::Relaxed);
        FORCE_PARALLEL_LOOP_FILTER.store(false, Ordering::Relaxed);
    }
}

#[test]
fn wavefront_loop_filter_matches_sequential_byte_for_byte() {
    // Repeat the parallel decode so a timing-dependent race has several chances to diverge from
    // the deterministic sequential reference.
    const PARALLEL_ITERS: usize = 5;
    let mut tested_vectors = 0;

    for name in HD_VECTORS {
        let Some(bytes) = common::read_vector(name) else {
            continue;
        };
        tested_vectors += 1;

        // Reference: the safe sequential loop filter.
        let sequential = {
            let _override = LoopFilterOverride::sequential();
            decode_displayed_frames(&bytes)
        };

        assert!(
            !sequential.is_empty(),
            "{name}: no displayed frames decoded (bad vector?)"
        );

        for iter in 0..PARALLEL_ITERS {
            let invocations_before = WAVEFRONT_INVOCATIONS.load(Ordering::Relaxed);
            let parallel = {
                let _override = LoopFilterOverride::parallel();
                decode_displayed_frames(&bytes)
            };
            let invocations_after = WAVEFRONT_INVOCATIONS.load(Ordering::Relaxed);
            assert!(
                invocations_after > invocations_before,
                "{name}: parallel iteration {iter} did not execute the wavefront"
            );
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

    if tested_vectors == 0 {
        eprintln!("[skip] loop-filter parallel vectors not found");
    }
}
