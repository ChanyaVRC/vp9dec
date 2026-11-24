//! Deblocking (loop) filter (spec §8.8 "Loop filter process").
//!
//! Spec §8.8's reference traversal of the frame is the following nested loops (taken directly from
//! the frame-wide traversal pseudocode at the top of the section); for parallelism the
//! implementation reorganizes it to plane-outermost, preserving each plane's order (see below):
//!
//! ```text
//! for ( row = 0; row < MiRows; row += 8 )
//!   for ( col = 0; col < MiCols; col += 8 )
//!     for ( plane = 0; plane < 3; plane++ )
//!       for ( pass = 0; pass < 2; pass++ )
//!         superblock loop filter process( plane, pass, row, col )
//! ```
//!
//! `pass == 0` means vertical edges (left/right block boundaries), and
//! `pass == 1` means horizontal edges (top/bottom block boundaries). Since
//! the same sample can be filtered multiple times, this traversal order
//! (vertical -> horizontal, superblock raster order) must be strictly
//! followed (see the NOTE in spec §8.8).
//!
//! For decode speed each plane is deblocked independently (the planes are disjoint buffers that
//! never read one another) and, above a size threshold, in parallel: within a plane the superblock
//! rows are filtered by a WAVEFRONT of worker threads (`wavefront_filter_planes`), each row
//! lagging the one above by two superblocks so their shared-corner writes stay ordered. This is
//! bit-exact -- the per-plane (row, col, pass) raster order above is preserved exactly; only
//! independent superblocks (and planes) ever run concurrently.
//!
//! # Known simplifications
//!
//! - `isIntra` (`RefFrames[row][col][0] <= INTRA_FRAME`) and `modeType`
//!   (whether `YModes` is `NEARESTMV`/`NEARMV`/`NEWMV`) are read from the
//!   real `ref_frame`/`y_mode` values added to `MiInfo` in M3 (see
//!   `superblock_loop_filter`).
//! - `loop_filter_ref_deltas`/`loop_filter_mode_deltas` persist across frames
//!   as spec §7.2 requires: `Decoder` (`lib.rs`) stores the previous frame's
//!   deltas and passes them into `parse_loop_filter_params`, which seeds from
//!   them (or from the default values `[1, 0, -1, -1]`/`[0, 0]` when
//!   `FrameIsIntra || error_resilient_mode`, i.e. `setup_past_independence`).

mod edge_filter;

use std::sync::atomic::{AtomicU32, Ordering};

use crate::common::{get_uv_tx_size, INTRA_FRAME};
use crate::framebuffer::Plane;
use crate::header::{LoopFilterParams, SegmentationParams};
use crate::prob_tables::{
    BLOCK_16X16, NEARESTMV, NEARMV, NEWMV, NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP,
};
use crate::tile::MiGrid;
#[cfg(test)]
use crate::header::SEG_LVL_ALT_L;
#[cfg(test)]
use crate::prob_tables::{TX_16X16, TX_4X4, TX_8X8};
#[cfg(test)]
use edge_filter::{adaptive_filter_strength, narrow_filter};
use edge_filter::{
    build_lvl_lookup, build_strength_table, filter_size_process, sample_filtering, LvlLookup,
    PlaneAccess, StrengthTable,
};

/// Minimum frame size (in MI units, `MiCols * MiRows`) at which [`loop_filter_frame`] filters
/// the planes on separate threads. Below it the per-frame synchronization outweighs the win,
/// so the planes are filtered sequentially on the calling thread. 4096 MI ~= 512x512 luma.
const LF_PARALLEL_MIN_MI: u64 = 4096;

/// Test-only knob: forces [`loop_filter_frame`] down the sequential path even when the intra-plane
/// wavefront would engage, so `tests/loop_filter_parallel_test.rs` can assert the parallel and
/// sequential loop filter of a real decode are byte-identical. Compiled only for test builds (the
/// self-referential `test-support` dev-dependency), mirroring `tile::FORCE_SEQUENTIAL_TILES`.
#[cfg(feature = "test-support")]
pub static FORCE_SEQUENTIAL_LOOP_FILTER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Test-only knob: forces eligible frames to use at least two wavefront workers even on a
/// single-logical-CPU test host. See [`FORCE_SEQUENTIAL_LOOP_FILTER`].
#[cfg(feature = "test-support")]
pub static FORCE_PARALLEL_LOOP_FILTER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Test-only count of calls to [`wavefront_filter_planes`], used to reject a vacuous
/// parallel-vs-sequential equality test.
#[cfg(feature = "test-support")]
pub static WAVEFRONT_INVOCATIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Raw-pointer view of one plane's pixel buffer for the multi-threaded loop-filter wavefront
/// ([`wavefront_filter_planes`]), where several worker threads write DISJOINT pixels of the *same*
/// plane at once. Rust forbids two `&mut Plane` (or overlapping `&mut [u16]`) to one buffer even
/// when the touched pixels are disjoint, so the workers share this `Copy` raw view instead; the
/// wavefront's per-row progress gate (a >=2 superblock-column lag with `Release`/`Acquire`) is what
/// actually keeps the accesses disjoint-or-ordered in time. The sequential path uses [`Plane`]
/// directly (fully safe) through the same [`PlaneAccess`] trait.
#[derive(Clone, Copy)]
struct PlaneView {
    ptr: *mut u16,
    stride: usize,
    x0: usize,
    height: usize,
}

// SAFETY: `PlaneView` is a bare pointer into a `Plane`'s `Vec<u16>`. It is only created from a
// `&mut Plane` that outlives the synchronous parallel dispatch owning the workers (so the buffer
// stays alive and is never moved/reallocated for the view's lifetime), and shared with workers
// that -- by the wavefront ordering in `wavefront_filter_planes` -- never write the same pixel
// concurrently and always establish happens-before (the `Release`/`Acquire` progress counter)
// before reading a pixel another worker wrote. Hence there is no data race.
unsafe impl Send for PlaneView {}
unsafe impl Sync for PlaneView {}

impl PlaneView {
    fn new(plane: &mut Plane) -> Self {
        let stride = plane.stride();
        let x0 = plane.x0;
        let height = plane.height;
        Self {
            ptr: plane.as_mut_slice().as_mut_ptr(),
            stride,
            x0,
            height,
        }
    }

    #[inline]
    fn offset(&self, x: usize, y: usize) -> usize {
        debug_assert!(
            x >= self.x0 && x - self.x0 < self.stride && y < self.height,
            "PlaneView access out of range"
        );
        y * self.stride + (x - self.x0)
    }
}

impl PlaneAccess for PlaneView {
    #[inline]
    fn lf_get(&self, x: usize, y: usize) -> u16 {
        // SAFETY: `offset` bounds-checks (debug). The wavefront guarantees any pixel this worker
        // reads is either its own or one another worker finished-and-published (Acquire) first, so
        // the read never races a concurrent write.
        unsafe { *self.ptr.add(self.offset(x, y)) }
    }
    #[inline]
    fn lf_set(&mut self, x: usize, y: usize, v: u16) {
        // SAFETY: `offset` bounds-checks (debug). The wavefront guarantees this worker has sole
        // access to (x, y) at this time (no other worker's superblock touches it concurrently).
        unsafe {
            *self.ptr.add(self.offset(x, y)) = v;
        }
    }
    #[inline]
    fn lf_base_ptr(&mut self) -> *mut u16 {
        self.ptr
    }
    #[inline]
    fn lf_stride(&self) -> usize {
        self.stride
    }
}

/// Per-8-lane-group filter gating/strength: `(active_lanes, apply_filter, lvl, filter_size)`.
///
/// Because `row`, `col`, and `i` are all multiples of 8, every 8-lane batch (`i..i+8`) along an
/// edge falls inside a single `(loop_row, loop_col)` MI unit for ALL VP9 chroma subsamplings
/// (`4:2:0`, `4:2:2`, `4:4:0`, `4:4:4`). Therefore `mi`, `lvl`, `filter_size`, `is_block_edge`,
/// and `is_tx_edge` are uniform across the active lanes; only the last superblock of an
/// odd-MI-dimension chroma plane can truncate `active_lanes` from 8 down to 4 (or 0).
#[inline]
#[allow(clippy::too_many_arguments)]
fn edge_group_params(
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lvl_lookup: &LvlLookup,
    plane_idx: usize,
    pass: u32,
    row: u32,
    col: u32,
    sub_x: u32,
    sub_y: u32,
    sub: u32,
    edge: u32,
    i: u32,
) -> (u32, u32, u32, bool, i32, u8) {
    let (x, y, active_lanes): (u32, u32, u32) = if pass == 0 {
        let x = col * 8 + edge * (4 << sub_x);
        let y = row * 8 + (i << sub_y);
        let max_y = 8 * mi_rows;
        if y >= max_y {
            return (x, y, 0, false, 0, 0);
        }
        let lanes = if y + (7 << sub_y) < max_y { 8 } else { 4 };
        (x, y, lanes)
    } else {
        let x = col * 8 + (i << sub_x);
        let y = row * 8 + edge * (4 << sub_y);
        let max_x = 8 * mi_cols;
        if x >= max_x {
            return (x, y, 0, false, 0, 0);
        }
        let lanes = if x + (7 << sub_x) < max_x { 8 } else { 4 };
        (x, y, lanes)
    };

    let loop_col = ((x >> 3) >> sub_x) << sub_x;
    let loop_row = ((y >> 3) >> sub_y) << sub_y;
    let mi = mi_grid.get(loop_row, loop_col);

    let mi_size = mi.mi_size;
    let tx_size = mi.tx_size;
    let tx_sz = if plane_idx > 0 {
        get_uv_tx_size(mi_size, tx_size, subsampling_x, subsampling_y)
    } else {
        tx_size
    };
    let sb_size = if sub == 0 {
        mi_size
    } else {
        mi_size.max(BLOCK_16X16)
    };
    let skip = mi.skip;
    let ref_frame = mi.ref_frame[0];
    let is_intra = ref_frame == INTRA_FRAME;

    // Block widths/heights in pixels (`8 * NUM_8X8_BLOCKS_*`) are powers of 2 -> bitmask test.
    let is_block_edge = if pass == 0 {
        (x & (8 * NUM_8X8_BLOCKS_WIDE_LOOKUP[sb_size as usize] as u32 - 1)) == 0
    } else {
        (y & (8 * NUM_8X8_BLOCKS_HIGH_LOOKUP[sb_size as usize] as u32 - 1)) == 0
    };

    // For pass == 1, sub_x == 1, odd mi_cols, and the last 4 active lanes (`x + 8 >= 8 * mi_cols`
    // holds for all 4 active lanes simultaneously because `x` is a multiple of 16 and `8 * mi_cols`
    // is `16k + 8`, so `x + 8 == 8 * mi_cols` on lane 0 and `> 8 * mi_cols` on lanes 1..3).
    let is_tx_edge =
        if pass == 1 && sub_x == 1 && (mi_cols & 1) == 1 && (edge & 1) == 1 && (x + 8) >= mi_cols * 8
        {
            false
        } else {
            (edge & ((1u32 << tx_sz) - 1)) == 0
        };

    let apply_filter = is_block_edge || (is_tx_edge && (is_intra || !skip));
    if !apply_filter {
        return (x, y, active_lanes, false, 0, 0);
    }

    let mode_type = matches!(mi.y_mode, NEARESTMV | NEARMV | NEWMV) as usize;
    let lvl = lvl_lookup[mi.segment_id as usize][ref_frame as usize][mode_type];
    if lvl <= 0 {
        return (x, y, active_lanes, false, 0, 0);
    }

    let is_32_edge = (edge & 7) == 0;
    let filter_size = filter_size_process(
        tx_sz, is_32_edge, pass, x, y, sub_x, sub_y, mi_cols, mi_rows,
    );

    (x, y, active_lanes, true, lvl, filter_size)
}

/// Spec §8.8.2 "Superblock loop filter process".
#[allow(clippy::too_many_arguments)]
fn superblock_loop_filter<P: PlaneAccess>(
    plane: &mut P,
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lvl_lookup: &LvlLookup,
    strength_table: &StrengthTable,
    plane_idx: usize,
    pass: u32,
    row: u32,
    col: u32,
    bit_depth: u8,
) {
    let (sub_x, sub_y) = if plane_idx == 0 {
        (0u32, 0u32)
    } else {
        (subsampling_x, subsampling_y)
    };

    let (dx, dy, sub, edge_len): (i64, i64, u32, u32) = if pass == 0 {
        (1, 0, sub_x, 64 >> sub_y)
    } else {
        (0, 1, sub_y, 64 >> sub_x)
    };

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::simd::avx2_enabled();
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let num_edges = 16u32 >> sub;
    for edge in 0..num_edges {
        // Edge-level screen boundary check: `x` (pass 0) or `y` (pass 1) is constant along the
        // entire edge, so skip edge 0 at frame origin and break past the frame's far edge.
        if pass == 0 {
            let edge_x = col * 8 + edge * (4 << sub_x);
            if edge_x == 0 {
                continue;
            }
            if edge_x >= 8 * mi_cols {
                break;
            }
        } else {
            let edge_y = row * 8 + edge * (4 << sub_y);
            if edge_y == 0 {
                continue;
            }
            if edge_y >= 8 * mi_rows {
                break;
            }
        }

        let mut i = 0u32;
        while i < edge_len {
            let (x, y, active_lanes, apply_filter, lvl, filter_size) = edge_group_params(
                mi_grid,
                mi_cols,
                mi_rows,
                subsampling_x,
                subsampling_y,
                lvl_lookup,
                plane_idx,
                pass,
                row,
                col,
                sub_x,
                sub_y,
                sub,
                edge,
                i,
            );
            if active_lanes == 0 {
                break;
            }
            if apply_filter {
                let (limit, blimit, thresh) = strength_table[lvl as usize];
                let px0 = (x >> sub_x) as usize;
                let py0 = (y >> sub_y) as usize;
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    let plane_width = plane.lf_stride();
                    let base = plane.lf_base_ptr();
                    if pass == 1 {
                        // SAFETY: `avx2_enabled()` confirmed; active lanes lie within the
                        // superblock-padded plane, and `filter_size == TX_16X16` guarantees the
                        // `p7..q7` window is in bounds (same proof as scalar `sample_filtering`).
                        unsafe {
                            crate::simd::loop_filter_horiz8_uniform_avx2(
                                base,
                                plane_width,
                                px0,
                                py0,
                                active_lanes,
                                filter_size,
                                limit,
                                blimit,
                                thresh,
                                bit_depth,
                            );
                        }
                    } else {
                        // SAFETY: transpose of the horizontal contract above.
                        unsafe {
                            crate::simd::loop_filter_vert8_uniform_avx2(
                                base,
                                plane_width,
                                px0,
                                py0,
                                active_lanes,
                                filter_size,
                                limit,
                                blimit,
                                thresh,
                                bit_depth,
                            );
                        }
                    }
                    i += 8;
                    continue;
                }

                let _ = use_avx2;
                for lane in 0..active_lanes as usize {
                    let (sx, sy) = if pass == 0 {
                        (px0, py0 + lane)
                    } else {
                        (px0 + lane, py0)
                    };
                    sample_filtering(
                        plane,
                        sx,
                        sy,
                        dx,
                        dy,
                        limit,
                        blimit,
                        thresh,
                        filter_size,
                        bit_depth,
                    );
                }
            }
            i += 8;
        }
    }
}

/// Filters one plane over the whole frame sequentially in the spec's per-plane order (superblock
/// raster; vertical then horizontal pass). The single-threaded fallback used by
/// [`loop_filter_frame`] for small frames / single-core machines; larger frames filter each plane
/// with [`wavefront_filter_planes`] instead.
#[allow(clippy::too_many_arguments)]
fn loop_filter_plane(
    plane: &mut Plane,
    plane_idx: usize,
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lvl_lookup: &LvlLookup,
    strength_table: &StrengthTable,
    bit_depth: u8,
) {
    let mut row = 0u32;
    while row < mi_rows {
        let mut col = 0u32;
        while col < mi_cols {
            for pass in 0..2u32 {
                superblock_loop_filter(
                    plane,
                    mi_grid,
                    mi_cols,
                    mi_rows,
                    subsampling_x,
                    subsampling_y,
                    lvl_lookup,
                    strength_table,
                    plane_idx,
                    pass,
                    row,
                    col,
                    bit_depth,
                );
            }
            col += 8;
        }
        row += 8;
    }
}

/// Upper bound on loop-filter wavefront threads. Because the 2-superblock-column lag limits
/// useful wavefront width and Windows scheduler contention escalates when spawning more spinning
/// threads than needed, capping at 8 keeps all cores productive without starvation stalls.
const MAX_LF_WAVEFRONT_THREADS: usize = 8;

/// Filters all three planes using the intra-plane WAVEFRONT: each of `n_threads` workers
/// processes a round-robin subset of superblock rows, in the spec's per-plane order (vertical
/// then horizontal pass per superblock), advancing through plane 0, then 1, then 2.
///
/// Uses `pool` (persistent worker threads owned by [`crate::Decoder`]) when supplied, falling
/// back to `std::thread::scope` for standalone callers/tests.
#[allow(clippy::too_many_arguments)]
fn wavefront_filter_planes(
    planes: &mut [Plane; 3],
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lvl_lookup: &LvlLookup,
    strength_table: &StrengthTable,
    bit_depth: u8,
    n_threads: usize,
    pool: Option<&mut crate::thread_pool::ThreadPool>,
) {
    #[cfg(feature = "test-support")]
    WAVEFRONT_INVOCATIONS.fetch_add(1, Ordering::Relaxed);

    let n_sb_rows = mi_rows.div_ceil(8) as usize;
    let n_sb_cols = mi_cols.div_ceil(8) as usize;
    // One `progress` array per plane: `progress[p][r]` = superblock columns fully filtered (both
    // passes) in row `r` of plane `p`. Published with `Release`, awaited with `Acquire`.
    let progress: [Vec<AtomicU32>; 3] =
        std::array::from_fn(|_| (0..n_sb_rows).map(|_| AtomicU32::new(0)).collect());
    let progress = &progress;
    let [p0, p1, p2] = planes;
    let views = [PlaneView::new(p0), PlaneView::new(p1), PlaneView::new(p2)];

    let run_worker = |t: usize| {
        // Advance through the planes in order; each is an independent wavefront, so a worker
        // that runs out of its rows in plane `p` flows straight into `p + 1` (no barrier).
        for plane_idx in 0..3usize {
            // Private `Copy` of this plane's shared raw view; writes go to the one
            // underlying buffer, kept disjoint-in-time by the gate below.
            let mut view = views[plane_idx];
            let prog = &progress[plane_idx];
            let mut sb_row = t;
            while sb_row < n_sb_rows {
                let row = (sb_row * 8) as u32;
                for sb_col in 0..n_sb_cols {
                    // Wavefront gate: wait until this plane's row above has finished column
                    // `sb_col + 1` (the 2-superblock lag). Row 0 has no row above.
                    // Spin briefly then yield if preempted, preventing scheduler starvation.
                    if sb_row > 0 {
                        let need = (sb_col as u32 + 2).min(n_sb_cols as u32);
                        let mut spins = 0u32;
                        while prog[sb_row - 1].load(Ordering::Acquire) < need {
                            if spins < 256 {
                                std::hint::spin_loop();
                                spins += 1;
                            } else {
                                std::thread::yield_now();
                                spins = 0;
                            }
                        }
                    }
                    let col = (sb_col * 8) as u32;
                    for pass in 0..2u32 {
                        superblock_loop_filter(
                            &mut view,
                            mi_grid,
                            mi_cols,
                            mi_rows,
                            subsampling_x,
                            subsampling_y,
                            lvl_lookup,
                            strength_table,
                            plane_idx,
                            pass,
                            row,
                            col,
                            bit_depth,
                        );
                    }
                    // Publish: this superblock (both passes) is done; unblock the row below.
                    prog[sb_row].store(sb_col as u32 + 1, Ordering::Release);
                }
                sb_row += n_threads;
            }
        }
    };

    if let Some(pool) = pool {
        pool.parallel_for(n_threads, run_worker);
    } else {
        std::thread::scope(|s| {
            for t in 1..n_threads {
                s.spawn(move || run_worker(t));
            }
            run_worker(0);
        });
    }
}

/// Frame-wide entry point for spec §8.8 "Loop filter process".
///
/// `planes` is `CurrFrame` (equivalent to `TileDecoder::planes`/`planes_mut`,
/// a buffer allocated out to superblock boundaries); `mi_grid` is the mode
/// info grid; `mi_cols`/`mi_rows` is the non-padded, mi-unit frame size
/// obtained from `compute_image_size()`.
#[allow(clippy::too_many_arguments)]
pub fn loop_filter_frame(
    planes: &mut [Plane; 3],
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
    bit_depth: u8,
) {
    loop_filter_frame_with_pool(
        planes,
        mi_grid,
        mi_cols,
        mi_rows,
        subsampling_x,
        subsampling_y,
        lf,
        seg,
        bit_depth,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn loop_filter_frame_with_pool(
    planes: &mut [Plane; 3],
    mi_grid: &MiGrid,
    mi_cols: u32,
    mi_rows: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
    bit_depth: u8,
    pool: Option<&mut crate::thread_pool::ThreadPool>,
) {
    let lvl_lookup = build_lvl_lookup(lf, seg);
    let strength_table = build_strength_table(lf.sharpness, bit_depth);

    #[cfg(feature = "test-support")]
    let force_sequential = FORCE_SEQUENTIAL_LOOP_FILTER.load(Ordering::Relaxed);
    #[cfg(not(feature = "test-support"))]
    let force_sequential = false;
    #[cfg(feature = "test-support")]
    let force_parallel = FORCE_PARALLEL_LOOP_FILTER.load(Ordering::Relaxed);
    #[cfg(not(feature = "test-support"))]
    let force_parallel = false;

    // Cheap gates first, so only frames that actually parallelize pay for the
    // `available_parallelism()` query; `n_threads == 1` then funnels into the sequential path below.
    let n_threads = if force_sequential
        || (!force_parallel && (mi_cols as u64) * (mi_rows as u64) < LF_PARALLEL_MIN_MI)
    {
        1
    } else {
        let n_sb_rows = mi_rows.div_ceil(8) as usize;
        let n_sb_cols = mi_cols.div_ceil(8) as usize;
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if force_parallel {
            available
                .max(2)
                .min(n_sb_rows.max(1))
                .min(MAX_LF_WAVEFRONT_THREADS)
        } else {
            let wavefront_cap = n_sb_rows
                .min(n_sb_cols.div_ceil(2))
                .min(MAX_LF_WAVEFRONT_THREADS);
            available.min(wavefront_cap.max(1))
        }
    };

    if n_threads <= 1 {
        for (plane_idx, plane) in planes.iter_mut().enumerate() {
            loop_filter_plane(
                plane,
                plane_idx,
                mi_grid,
                mi_cols,
                mi_rows,
                subsampling_x,
                subsampling_y,
                &lvl_lookup,
                &strength_table,
                bit_depth,
            );
        }
        return;
    }

    wavefront_filter_planes(
        planes,
        mi_grid,
        mi_cols,
        mi_rows,
        subsampling_x,
        subsampling_y,
        &lvl_lookup,
        &strength_table,
        bit_depth,
        n_threads,
        pool,
    );
}

#[cfg(test)]
#[path = "../tests/unit/loop_filter.rs"]
mod tests;
