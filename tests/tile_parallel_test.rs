//! Tile-parallel decode must be byte-identical to the sequential tile loop.
//!
//! The official multi-tile vectors (`vp90-2-08-tile_1x{2,4,8}`) normally take the tile-parallel
//! fast path (`tile::decode_tiles_parallel`, engaged for >1 tile column and 1 tile row). This
//! test decodes each vector twice -- once normally (parallel) and once with the test-only
//! `FORCE_SEQUENTIAL_TILES` knob -- and requires every displayed frame's planes to match
//! exactly, pinning the column-strip worker buffers + merge against the sequential reference.
//! Skips cleanly if the vectors haven't been downloaded.

mod common;

use std::sync::atomic::Ordering;

use vp9dec::ivf::IvfReader;
use vp9dec::tile::FORCE_SEQUENTIAL_TILES;
use vp9dec::Decoder;

const TILE_VECTORS: [&str; 3] = [
    "vp90-2-08-tile_1x2.ivf",
    "vp90-2-08-tile_1x4.ivf",
    "vp90-2-08-tile_1x8.ivf",
];

/// Decodes the whole IVF stream and returns each displayed frame's planes as I420 bytes.
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
fn parallel_tile_decode_matches_sequential_byte_for_byte() {
    for name in TILE_VECTORS {
        let Some(bytes) = common::read_vector(name) else {
            continue;
        };

        let parallel = decode_displayed_frames(&bytes);

        FORCE_SEQUENTIAL_TILES.store(true, Ordering::Relaxed);
        let sequential = decode_displayed_frames(&bytes);
        FORCE_SEQUENTIAL_TILES.store(false, Ordering::Relaxed);

        assert_eq!(
            parallel.len(),
            sequential.len(),
            "{name}: displayed frame count differs"
        );
        for (i, (p, s)) in parallel.iter().zip(sequential.iter()).enumerate() {
            assert_eq!(p, s, "{name}: frame {i} differs (parallel vs sequential)");
        }
        eprintln!(
            "{name}: {} displayed frames byte-identical parallel vs sequential",
            parallel.len()
        );
    }
}
