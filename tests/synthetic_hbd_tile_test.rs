//! End-to-end coverage for the corpus-unreachable combination of profile-3 high-bit-depth
//! 4:4:4 output and tile-column worker strips.
//!
//! Official HBD vectors are only about 160 pixels wide. VP9 cannot signal two tile columns
//! until a frame has at least eight 64x64 superblock columns (449 pixels), so none reaches the
//! parallel tile path. This hand-built 512x64 key frame has two 256-pixel tile columns and uses
//! distinct edge-derived intra predictions in them: V_PRED produces 511 in the left tile and
//! H_PRED produces 513 in the right tile at 10-bit depth. That gives every Y/U/V plane a
//! non-vacuous, exact oracle at the strip merge boundary.

mod common;

use std::process::Command;
use std::sync::atomic::Ordering;

use common::encoder::{
    assemble_frame, build_keyframe_compressed_header, build_keyframe_header_with_config,
    encode_tree, header_size, KeyframeConfig, SegSpec,
};
use vp9dec::header::{parse_uncompressed_header, FrameHeader, PersistentState};
use vp9dec::prob_tables::{
    DC_PRED, DEFAULT_SKIP_PROB, H_PRED, INTRA_MODE_TREE, KF_PARTITION_PROBS, KF_UV_MODE_PROBS,
    KF_Y_MODE_PROBS, PARTITION_NONE, PARTITION_TREE, V_PRED,
};
use vp9dec::test_support::BoolEncoder;
use vp9dec::tile::FORCE_TILE_WORKERS;
use vp9dec::{Decoder, Frame, PlaneData};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 64;
const TILE_BOUNDARY_X: usize = 256;
const LEFT_VALUE: u16 = 511;
const RIGHT_VALUE: u16 = 513;

/// One 256x64 tile: four 64x64 `PARTITION_NONE`, lossless/skip blocks in one row.
///
/// Partition context stays 12 because each 64x64 NONE update leaves the relevant context bit
/// clear. Skip context is 0 for the tile's first block and 1 afterward (left block is skip=1);
/// above mode is always DC_PRED, while the left mode becomes `mode` after the first block.
fn encode_uniform_intra_tile(mode: u8) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    for sb_col in 0..4 {
        encode_tree(&mut enc, &PARTITION_TREE, PARTITION_NONE, |node| {
            KF_PARTITION_PROBS[12][node]
        });

        let skip_ctx = usize::from(sb_col != 0);
        enc.write_bool(true, DEFAULT_SKIP_PROB[skip_ctx]);

        let left_mode = if sb_col == 0 { DC_PRED } else { mode };
        encode_tree(&mut enc, &INTRA_MODE_TREE, mode, |node| {
            KF_Y_MODE_PROBS[DC_PRED as usize][left_mode as usize][node]
        });
        encode_tree(&mut enc, &INTRA_MODE_TREE, mode, |node| {
            KF_UV_MODE_PROBS[mode as usize][node]
        });
    }
    enc.finish()
}

fn build_profile3_two_tile_frame() -> Vec<u8> {
    let compressed = build_keyframe_compressed_header();
    let header = build_keyframe_header_with_config(
        WIDTH,
        HEIGHT,
        0,
        false,
        &SegSpec::disabled(),
        header_size(&compressed),
        KeyframeConfig {
            profile: 3,
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            tile_cols_log2: 1,
        },
    );

    let mut left = encode_uniform_intra_tile(V_PRED);
    let mut right = encode_uniform_intra_tile(H_PRED);
    // A conformant bool-coded partition may pull a few zero bits while finalizing its range.
    // This decoder models reads beyond the provided slice as zero, but external decoders flag
    // that as truncation, so carry an explicit zero tail inside each tile's declared size.
    left.extend_from_slice(&[0; 4]);
    right.extend_from_slice(&[0; 4]);
    let mut tiles = Vec::with_capacity(4 + left.len() + right.len());
    tiles.extend_from_slice(
        &u32::try_from(left.len())
            .expect("synthetic tile size fits u32")
            .to_be_bytes(),
    );
    tiles.extend_from_slice(&left);
    tiles.extend_from_slice(&right);
    assemble_frame(header, compressed, tiles)
}

fn decode_single(frame_bytes: &[u8]) -> Frame {
    let decoded = Decoder::new()
        .decode_frame(frame_bytes)
        .expect("synthetic profile-3 multi-tile frame should decode");
    assert_eq!(decoded.len(), 1, "expected one constituent VP9 frame");
    decoded
        .into_iter()
        .next()
        .and_then(|decoded| decoded.frame)
        .expect("key frame should be shown")
}

fn assert_exact_split(plane: &PlaneData, name: &str) {
    let PlaneData::U16(samples) = plane else {
        panic!("{name}: profile-3 10-bit output must use PlaneData::U16");
    };
    assert_eq!(
        samples.len(),
        (WIDTH * HEIGHT) as usize,
        "{name}: unexpected 4:4:4 plane size"
    );
    for (y, row) in samples.chunks_exact(WIDTH as usize).enumerate() {
        assert!(
            row[..TILE_BOUNDARY_X]
                .iter()
                .all(|&sample| sample == LEFT_VALUE),
            "{name}: row {y} left tile is not uniformly {LEFT_VALUE}"
        );
        assert!(
            row[TILE_BOUNDARY_X..]
                .iter()
                .all(|&sample| sample == RIGHT_VALUE),
            "{name}: row {y} right tile is not uniformly {RIGHT_VALUE}"
        );
    }
}

/// Resets the test-only global even if an assertion panics.
struct TileWorkerReset;

impl Drop for TileWorkerReset {
    fn drop(&mut self) {
        FORCE_TILE_WORKERS.store(0, Ordering::Relaxed);
    }
}

fn probe_ffmpeg() -> Option<String> {
    let explicit = std::env::var("VP9DEC_FFMPEG").ok();
    let ffmpeg = explicit.clone().unwrap_or_else(|| "ffmpeg".to_string());
    let found = Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .is_ok_and(|out| out.status.success());
    assert!(
        found || explicit.is_none(),
        "VP9DEC_FFMPEG is set ({ffmpeg:?}) but does not run as ffmpeg"
    );
    found.then_some(ffmpeg)
}

fn available_decoders(ffmpeg: &str) -> Vec<&'static str> {
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-decoders"])
        .output()
        .expect("run ffmpeg -decoders");
    let listing = String::from_utf8_lossy(&out.stdout);
    ["libvpx-vp9", "vp9"]
        .into_iter()
        .filter(|decoder| {
            listing
                .lines()
                .any(|line| line.split_whitespace().nth(1) == Some(*decoder))
        })
        .collect()
}

fn cross_decode_with_ffmpeg(frame_bytes: &[u8], ours: &[u8]) {
    let Some(ffmpeg) = probe_ffmpeg() else {
        eprintln!(
            "[skip] ffmpeg not found; profile-3 multi-tile parallel/sequential checks still ran"
        );
        return;
    };
    let decoders = available_decoders(&ffmpeg);
    if decoders.is_empty() {
        eprintln!("[skip] {ffmpeg:?} provides neither libvpx-vp9 nor native vp9");
        return;
    }
    if decoders.len() < 2 {
        eprintln!(
            "[note] this ffmpeg build only provides {decoders:?}; cross-decoding with it alone"
        );
    }

    struct CleanOnDrop(std::path::PathBuf);
    impl Drop for CleanOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let tmp_dir =
        std::env::temp_dir().join(format!("vp9dec_hbd_tile_xdecode_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("create cross-decode temp dir");
    let _cleanup = CleanOnDrop(tmp_dir.clone());

    let ivf_path = tmp_dir.join("profile3_10bit_444_2tiles.ivf");
    std::fs::write(
        &ivf_path,
        vp9dec::ivf::write_ivf(
            b"VP90",
            WIDTH as u16,
            HEIGHT as u16,
            30,
            1,
            &[frame_bytes.to_vec()],
        ),
    )
    .expect("write synthetic IVF");

    for decoder in decoders {
        let out_path = tmp_dir.join(format!("{decoder}.yuv"));
        let output = Command::new(&ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-c:v",
                decoder,
                "-i",
            ])
            .arg(&ivf_path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv444p10le"])
            .arg(&out_path)
            .output()
            .expect("run ffmpeg cross-decode");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "ffmpeg -c:v {decoder} failed decoding synthetic HBD tile IVF:\n{stderr}"
        );
        assert!(
            stderr.trim().is_empty(),
            "ffmpeg -c:v {decoder} reported decode errors:\n{stderr}"
        );

        let external = std::fs::read(&out_path).expect("read ffmpeg raw output");
        assert_eq!(
            external, ours,
            "profile-3 multi-tile output differs from ffmpeg -c:v {decoder}"
        );
        eprintln!("[xdecode] profile3_10bit_444_2tiles/{decoder}: OK (U16 planes byte-identical)");
    }
}

#[test]
fn profile3_hbd_multi_tile_is_exact_parallel_sequential_and_cross_decoded() {
    let frame_bytes = build_profile3_two_tile_frame();
    let (parsed, _) = parse_uncompressed_header(&frame_bytes, &PersistentState::default())
        .expect("synthetic uncompressed header should parse");
    let FrameHeader::New(header) = parsed else {
        panic!("synthetic frame unexpectedly used show_existing_frame");
    };
    let color = header
        .color_config
        .expect("key frame must carry a color config");
    assert_eq!(
        (
            header.profile,
            color.bit_depth,
            color.subsampling_x,
            color.subsampling_y,
            header.tile_cols_log2,
            header.tile_rows_log2,
        ),
        (3, 10, 0, 0, 1, 0),
        "stream must actually select profile-3 10-bit 4:4:4 with two columns and one row"
    );

    let _reset = TileWorkerReset;
    FORCE_TILE_WORKERS.store(2, Ordering::Relaxed);
    let parallel = decode_single(&frame_bytes);
    FORCE_TILE_WORKERS.store(1, Ordering::Relaxed);
    let sequential = decode_single(&frame_bytes);

    assert_eq!(
        parallel, sequential,
        "two-worker tile decode differs from the one-worker sequential fallback"
    );
    assert_eq!(
        (
            parallel.width,
            parallel.height,
            parallel.bit_depth,
            parallel.subsampling_x,
            parallel.subsampling_y,
        ),
        (WIDTH, HEIGHT, 10, 0, 0)
    );
    assert_exact_split(&parallel.y, "Y");
    assert_exact_split(&parallel.u, "U");
    assert_exact_split(&parallel.v, "V");

    let ours = common::i420_bytes(&parallel);
    assert_eq!(
        ours.len(),
        (WIDTH * HEIGHT * 3 * 2) as usize,
        "10-bit 4:4:4 raw output is three 16-bit planes"
    );
    cross_decode_with_ffmpeg(&frame_bytes, &ours);
}
