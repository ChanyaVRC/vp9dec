//! Tile-parallel decode machinery (the `tile_cols > 1 && tile_rows == 1` fast path of
//! [`TileDecoder::decode_tiles`]): pooled per-tile-column worker `TileDecoder`s (column-strip
//! buffers), parallel disjoint strip merge back into the frame decoder, the tile-data splitter,
//! and the persistent-[`crate::thread_pool::ThreadPool`] (or `std::thread::scope` fallback)
//! driver. Split out of `tile.rs` (which keeps the sequential tile/superblock traversal this
//! path is bit-identical to).

use std::sync::{Arc, OnceLock};

use crate::compressed_header::CompressedHeaderProbs;
use crate::counts::Counts;
use crate::framebuffer::Plane;
use crate::thread_pool::ThreadPool;

use super::{get_tile_offset, MiGrid, MiInfo, TileDecoder, TileError};

/// Shared empty `prev_segment_ids` sentinel used when parking a worker `TileDecoder`, avoiding a
/// fresh heap allocation of an `Arc<Vec<u8>>` control block on every worker per frame.
fn empty_segment_ids() -> Arc<Vec<u8>> {
    static EMPTY: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
    Arc::clone(EMPTY.get_or_init(|| Arc::new(Vec::new())))
}

/// Shared default `probs` sentinel used when parking a worker `TileDecoder` so the frame's
/// `CompressedHeaderProbs` `Arc` is released before DPB/context cleanup.
fn default_probs_sentinel() -> Arc<CompressedHeaderProbs> {
    static DEFAULT: OnceLock<Arc<CompressedHeaderProbs>> = OnceLock::new();
    Arc::clone(DEFAULT.get_or_init(|| Arc::new(CompressedHeaderProbs::default())))
}

/// Test-only knob: forces [`TileDecoder::decode_tiles`] down the sequential loop even when the
/// tile-parallel fast path would engage, so tests can assert that the parallel and sequential
/// decodes of a multi-tile stream are byte-identical (see `tests/tile_parallel_test.rs`).
/// Compiled only for test builds (the self-referential `test-support` dev-dependency).
#[cfg(feature = "test-support")]
pub static FORCE_SEQUENTIAL_TILES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Test/benchmark-only override for the worker count reported to tile dispatch. Zero keeps the
/// platform's [`std::thread::available_parallelism`] result; a positive value makes the
/// single-worker fallback and bounded parallel cases reproducible on any development machine.
#[cfg(feature = "test-support")]
pub static FORCE_TILE_WORKERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(super) fn available_tile_workers() -> usize {
    #[cfg(feature = "test-support")]
    {
        let forced = FORCE_TILE_WORKERS.load(std::sync::atomic::Ordering::Relaxed);
        if forced > 0 {
            return forced;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// Raw pointer wrapper allowing each `parallel_for` worker `i` to access its own unique element
/// `ptr.add(i)` without creating overlapping `&mut` slices.
#[derive(Clone, Copy)]
struct DisjointSlicePtr<T> {
    ptr: *mut T,
    len: usize,
}
// SAFETY: Used exclusively inside `parallel_for(len, |i| ...)` where each task index `i` is
// invoked once on a backing slice of length `len` that outlives the synchronous dispatch.
unsafe impl<T: Send> Send for DisjointSlicePtr<T> {}
// SAFETY: Concurrent tasks only ever access strictly disjoint indices `i` (`0..len`), so
// sharing `&DisjointSlicePtr<T>` across workers produces no aliased `&mut T` references.
unsafe impl<T: Send> Sync for DisjointSlicePtr<T> {}

impl<T> DisjointSlicePtr<T> {
    #[inline]
    fn from_slice(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Returns a mutable pointer to element `i`.
    ///
    /// # Safety
    /// Caller must ensure `i < self.len`, the backing slice is still alive, and no other thread
    /// accesses element `i` concurrently.
    #[inline]
    unsafe fn add(&self, i: usize) -> *mut T {
        debug_assert!(i < self.len, "DisjointSlicePtr index out of bounds");
        // SAFETY: caller guarantees `i < self.len` and exclusive access to slot `i`.
        unsafe { self.ptr.add(i) }
    }
}

/// Raw write-only view of the destination frame's `MiGrid` and `[Plane; 3]` for parallel
/// disjoint column-strip merging.
#[derive(Clone, Copy)]
struct DstFrameView {
    mi_ptr: *mut MiInfo,
    mi_cols: usize,
    mi_rows: usize,
    plane_ptrs: [*mut u16; 3],
    plane_strides: [usize; 3],
    plane_heights: [usize; 3],
    subsampling_x: u32,
}
// SAFETY: Backed by `&mut TileDecoder`'s `mi_grid` and `planes` for the duration of the
// synchronous parallel dispatch; each worker writes only its own non-overlapping column range
// `[mi_col_start, mi_col_end)` (and corresponding `[px0, px1)` per plane).
unsafe impl Send for DstFrameView {}
// SAFETY: Sharing `&DstFrameView` across tile-column workers is sound because every worker's
// column range `[mi_col_start, mi_col_end)` is strictly disjoint.
unsafe impl Sync for DstFrameView {}

impl DstFrameView {
    fn new(dst: &mut TileDecoder) -> Self {
        let mi_cols = dst.mi_grid.cols();
        let mi_rows = dst.planes[0].height / 8;
        let mi_ptr = dst.mi_grid.data.as_mut_ptr();
        let [p0, p1, p2] = &mut dst.planes;
        let plane_strides = [p0.stride(), p1.stride(), p2.stride()];
        let plane_heights = [p0.height, p1.height, p2.height];
        let plane_ptrs = [
            p0.as_mut_slice().as_mut_ptr(),
            p1.as_mut_slice().as_mut_ptr(),
            p2.as_mut_slice().as_mut_ptr(),
        ];
        Self {
            mi_ptr,
            mi_cols,
            mi_rows,
            plane_ptrs,
            plane_strides,
            plane_heights,
            subsampling_x: dst.subsampling_x,
        }
    }

    /// Copies worker `w`'s disjoint column strip of `mi_grid` and `planes` into the destination
    /// frame buffer.
    ///
    /// # Safety
    /// Caller must ensure `[w.mi_col_start, w.mi_col_end)` is disjoint from every other
    /// concurrently executing worker's column range and within the destination frame's bounds.
    unsafe fn copy_worker_strip(&self, w: &TileDecoder) {
        let (c0, c1) = (w.mi_col_start as usize, w.mi_col_end as usize);
        let strip_cols = c1 - c0;
        let src_cols = w.mi_grid.cols();
        let src_mi = w.mi_grid.data.as_ptr();
        for row in 0..self.mi_rows {
            // SAFETY: `[c0, c1)` is disjoint per worker and `< self.mi_cols`; `row < self.mi_rows`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_mi.add(row * src_cols),
                    self.mi_ptr.add(row * self.mi_cols + c0),
                    strip_cols,
                );
            }
        }
        for p in 0..3 {
            let sub = if p == 0 { 0 } else { self.subsampling_x };
            let px0 = (c0 * 8) >> sub;
            let px1 = (c1 * 8) >> sub;
            let copy_w = px1 - px0;
            let dst_stride = self.plane_strides[p];
            let height = self.plane_heights[p];
            let src_stride = w.planes[p].stride();
            let src_ptr = w.planes[p].as_slice().as_ptr();
            let dst_ptr = self.plane_ptrs[p];
            debug_assert_eq!(w.planes[p].x0, px0);
            for row in 0..height {
                // SAFETY: `[px0, px1)` is disjoint per worker and within row stride; `row < height`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(row * src_stride),
                        dst_ptr.add(row * dst_stride + px0),
                        copy_w,
                    );
                }
            }
        }
    }
}

/// Precomputed column-strip dimensions for a tile-column worker covering `[mi_col_start, mi_col_end)`.
struct ColumnStripGeometry {
    grid_cols: usize,
    grid_rows: usize,
    above_nz_len: usize,
    strip_mi_cols: usize,
    plane_strips: [(usize, usize, usize); 3], // (x0, width, height)
}

impl ColumnStripGeometry {
    fn new(parent: &TileDecoder, mi_col_start: u32, mi_col_end: u32) -> Self {
        let grid_cols = parent.mi_grid.cols();
        let grid_rows = parent.planes[0].height / 8;
        let last = mi_col_end == parent.mi_cols;
        let grid_col_end = if last { grid_cols } else { mi_col_end as usize };
        let plane_strips = std::array::from_fn(|p| {
            let sub = if p == 0 { 0 } else { parent.subsampling_x };
            let x0 = (mi_col_start as usize * 8) >> sub;
            let x_end = if last {
                parent.planes[p].width
            } else {
                (mi_col_end as usize * 8) >> sub
            };
            (x0, x_end - x0, parent.planes[p].height)
        });
        Self {
            grid_cols,
            grid_rows,
            above_nz_len: grid_cols * 2,
            strip_mi_cols: grid_col_end - mi_col_start as usize,
            plane_strips,
        }
    }
}

impl TileDecoder {
    /// Copies every read-only frame configuration field and cheap `Arc` handle from `parent`
    /// onto this worker and binds its active tile-column range `[mi_col_start, mi_col_end)`.
    /// Shared by [`Self::spawn_column_worker`] and [`Self::reuse_or_spawn_column_worker`] so the
    /// two paths can never drift when a field is added to [`TileDecoder`].
    fn sync_frame_config_from(&mut self, parent: &TileDecoder, mi_col_start: u32, mi_col_end: u32) {
        self.tx_mode = parent.tx_mode;
        self.probs = parent.probs.clone();
        self.segmentation = parent.segmentation;
        self.mi_cols = parent.mi_cols;
        self.mi_rows = parent.mi_rows;
        self.tile_cols_log2 = parent.tile_cols_log2;
        self.tile_rows_log2 = parent.tile_rows_log2;
        self.prev_segment_ids = parent.prev_segment_ids.clone();
        self.mi_col_start = mi_col_start;
        self.mi_col_end = mi_col_end;
        self.mi_row_start = 0;
        self.mi_row_end = parent.mi_rows;
        self.bit_depth = parent.bit_depth;
        self.subsampling_x = parent.subsampling_x;
        self.subsampling_y = parent.subsampling_y;
        self.lossless = parent.lossless;
        self.dequant_table = parent.dequant_table;
        self.frame_is_intra = parent.frame_is_intra;
        self.ref_frame_sign_bias = parent.ref_frame_sign_bias;
        self.allow_high_precision_mv = parent.allow_high_precision_mv;
        self.interpolation_filter = parent.interpolation_filter;
        self.reference_mode = parent.reference_mode;
        self.comp_fixed_ref = parent.comp_fixed_ref;
        self.comp_var_ref = parent.comp_var_ref;
        self.use_prev_frame_mvs = parent.use_prev_frame_mvs;
        self.prev_mi_grid = parent.prev_mi_grid.clone();
        self.frame_width = parent.frame_width;
        self.frame_height = parent.frame_height;
        self.resolved_refs = parent.resolved_refs.clone();
        self.collect_counts = parent.collect_counts;
    }

    /// Builds a fresh worker `TileDecoder` that shares this frame's read-only / config state
    /// (cheap `Arc` / `Copy` clones) but gets its own mutable buffers, sized to the tile
    /// column `[mi_col_start, mi_col_end)` it will decode: the planes and `mi_grid` are *column
    /// strips* (see [`Plane::new_strip`] / [`MiGrid::new_strip`] -- indexed by absolute
    /// coordinates, so the decode path is unchanged), not full frames.
    pub(super) fn spawn_column_worker(&self, mi_col_start: u32, mi_col_end: u32) -> TileDecoder {
        let geom = ColumnStripGeometry::new(self, mi_col_start, mi_col_end);
        let planes = std::array::from_fn(|p| {
            let (x0, w, h) = geom.plane_strips[p];
            Plane::new_strip(w, h, x0)
        });
        let mut worker = TileDecoder {
            tx_mode: self.tx_mode,
            probs: self.probs.clone(),
            segmentation: self.segmentation,
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            tile_cols_log2: self.tile_cols_log2,
            tile_rows_log2: self.tile_rows_log2,
            mi_grid: MiGrid::new_strip(geom.strip_mi_cols, geom.grid_rows, mi_col_start),
            above_partition_context: vec![0u8; geom.grid_cols],
            left_partition_context: [0u8; 8],
            above_seg_pred_context: vec![0u8; geom.grid_cols],
            left_seg_pred_context: [0u8; 8],
            prev_segment_ids: self.prev_segment_ids.clone(),
            mi_col_start,
            mi_col_end,
            mi_row_start: 0,
            mi_row_end: self.mi_rows,
            bit_depth: self.bit_depth,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            lossless: self.lossless,
            dequant_table: self.dequant_table,
            planes,
            above_nonzero_context: [
                vec![0u8; geom.above_nz_len],
                vec![0u8; geom.above_nz_len],
                vec![0u8; geom.above_nz_len],
            ],
            left_nonzero_context: [[0u8; 16]; 3],
            frame_is_intra: self.frame_is_intra,
            ref_frame_sign_bias: self.ref_frame_sign_bias,
            allow_high_precision_mv: self.allow_high_precision_mv,
            interpolation_filter: self.interpolation_filter,
            reference_mode: self.reference_mode,
            comp_fixed_ref: self.comp_fixed_ref,
            comp_var_ref: self.comp_var_ref,
            use_prev_frame_mvs: self.use_prev_frame_mvs,
            prev_mi_grid: self.prev_mi_grid.clone(),
            frame_width: self.frame_width,
            frame_height: self.frame_height,
            resolved_refs: self.resolved_refs.clone(),
            counts: Counts::new(),
            collect_counts: self.collect_counts,
            dequant_buf: [0i64; 1024],
            token_cache_buf: [0u8; 1024],
        };
        worker.sync_frame_config_from(self, mi_col_start, mi_col_end);
        worker
    }

    /// Reuses a pooled column-worker `TileDecoder` when available (reusing its strip `planes`,
    /// `mi_grid`, and context allocations without heap alloc/free), or spawns a fresh one.
    fn reuse_or_spawn_column_worker(
        &self,
        mi_col_start: u32,
        mi_col_end: u32,
        recycled: Option<TileDecoder>,
    ) -> TileDecoder {
        let Some(mut w) = recycled else {
            return self.spawn_column_worker(mi_col_start, mi_col_end);
        };

        let geom = ColumnStripGeometry::new(self, mi_col_start, mi_col_end);
        if !w
            .mi_grid
            .try_reuse_as_strip(geom.strip_mi_cols, geom.grid_rows, mi_col_start)
        {
            w.mi_grid = MiGrid::new_strip(geom.strip_mi_cols, geom.grid_rows, mi_col_start);
        }
        for (p, &(x0, strip_w, strip_h)) in geom.plane_strips.iter().enumerate() {
            if !w.planes[p].try_reuse_as_strip(strip_w, strip_h, x0) {
                w.planes[p] = Plane::new_strip(strip_w, strip_h, x0);
            }
        }

        w.above_partition_context.resize(geom.grid_cols, 0);
        w.above_partition_context.fill(0);
        w.above_seg_pred_context.resize(geom.grid_cols, 0);
        w.above_seg_pred_context.fill(0);
        for ctx in &mut w.above_nonzero_context {
            ctx.resize(geom.above_nz_len, 0);
            ctx.fill(0);
        }
        w.clear_left_context();

        w.sync_frame_config_from(self, mi_col_start, mi_col_end);
        if self.collect_counts {
            w.counts = Counts::new();
        }
        w
    }

    /// Drops frame-specific `Arc` handles (`probs`, `resolved_refs`, `prev_mi_grid`,
    /// `prev_segment_ids`) before parking a worker back in `Decoder::tile_worker_pool`, so DPB,
    /// `MiGrid`, and `CompressedHeaderProbs` recycling observes `Arc::strong_count == 1`.
    fn release_frame_refs(&mut self) {
        self.probs = default_probs_sentinel();
        self.resolved_refs = [None, None, None];
        self.prev_mi_grid = None;
        self.prev_segment_ids = empty_segment_ids();
    }

    /// Merges tile-column worker `w` (which decoded MI columns `[w.mi_col_start, w.mi_col_end)`
    /// into column-strip buffers) back into `self`: copies that column strip of every plane and
    /// of `mi_grid`, and sums the worker's probability-adaptation counts. Kept for unit tests.
    #[cfg(test)]
    pub(super) fn merge_column_worker(&mut self, w: &TileDecoder) {
        if self.collect_counts {
            self.counts.add_assign(&w.counts);
        }
        let dst_view = DstFrameView::new(self);
        // SAFETY: sole writer to `self`; `[w.mi_col_start, w.mi_col_end)` is in bounds.
        unsafe {
            dst_view.copy_worker_strip(w);
        }
    }

    /// Splits `data` (the concatenated tiles, each non-final one prefixed by a 4-byte big-endian
    /// size) into the `num_tiles` per-tile byte slices. Shared by the sequential loop in
    /// [`Self::decode_tiles`] (which iterates the returned slices) and the parallel path (which
    /// hands them to worker threads).
    pub(super) fn split_tiles(mut data: &[u8], num_tiles: usize) -> Result<Vec<&[u8]>, TileError> {
        let mut out = Vec::with_capacity(num_tiles);
        for i in 0..num_tiles {
            let last = i == num_tiles - 1;
            let size = if last {
                data.len()
            } else {
                let Some((size_bytes, rest)) = data.split_first_chunk::<4>() else {
                    return Err(TileError::InvalidTileSize);
                };
                data = rest;
                u32::from_be_bytes(*size_bytes) as usize
            };
            if data.len() < size {
                return Err(TileError::InvalidTileSize);
            }
            let (tile_bytes, rest) = data.split_at(size);
            out.push(tile_bytes);
            data = rest;
        }
        Ok(out)
    }

    /// Tile-parallel `decode_tiles` for the `tile_cols > 1 && tile_rows == 1` case: decode each
    /// tile column on its own worker thread (VP9 tile columns are independent) and merge its
    /// disjoint column strip back into `self` in parallel. Bit-identical to the sequential path
    /// (disjoint column writes; order-independent count sums).
    pub(super) fn decode_tiles_parallel(
        &mut self,
        data: &[u8],
        tile_cols: u32,
        max_workers: usize,
        mut pool: Option<&mut ThreadPool>,
        mut worker_pool: Option<&mut Vec<TileDecoder>>,
    ) -> Result<(), TileError> {
        let tiles = Self::split_tiles(data, tile_cols as usize)?;

        debug_assert!(max_workers > 1);
        let chunk = max_workers.min(tile_cols as usize) as u32;
        let mut parked: Vec<TileDecoder> = worker_pool
            .as_deref_mut()
            .map(std::mem::take)
            .unwrap_or_default();

        let mut c0 = 0u32;
        let mut first_err: Option<TileError> = None;
        while c0 < tile_cols {
            let c1 = (c0 + chunk).min(tile_cols);
            let n_chunk = (c1 - c0) as usize;
            let mut existing = std::mem::take(&mut parked).into_iter();
            let mut workers: Vec<TileDecoder> = (c0..c1)
                .map(|c| {
                    self.reuse_or_spawn_column_worker(
                        get_tile_offset(c, self.mi_cols, self.tile_cols_log2),
                        get_tile_offset(c + 1, self.mi_cols, self.tile_cols_log2),
                        existing.next(),
                    )
                })
                .collect();

            let mut results: Vec<(
                Result<(), TileError>,
                [u64; crate::bench_timing::STAGE_COUNT],
            )> = vec![(Ok(()), [0u64; crate::bench_timing::STAGE_COUNT]); n_chunk];

            let tile_slice = &tiles[c0 as usize..c1 as usize];
            let dst_view = DstFrameView::new(self);
            let workers_ptr = DisjointSlicePtr::from_slice(&mut workers);
            let results_ptr = DisjointSlicePtr::from_slice(&mut results);

            let run_task = |i: usize| {
                // SAFETY: each `i in 0..n_chunk` runs once, so `workers_ptr.add(i)` and
                // `results_ptr.add(i)` are exclusively owned by this task, and `w`'s tile
                // column range `[mi_col_start, mi_col_end)` is disjoint from all other tasks.
                unsafe {
                    let w = &mut *workers_ptr.add(i);
                    // Task 0 runs on the calling thread, which already accumulates directly into
                    // the caller's `bench_timing` counters; only worker threads reset/snapshot.
                    if i != 0 {
                        crate::bench_timing::reset();
                    }
                    let res = w.decode_tile_bytes(tile_slice[i]);
                    if res.is_ok() {
                        dst_view.copy_worker_strip(w);
                    }
                    let snap = if i != 0 {
                        crate::bench_timing::snapshot()
                    } else {
                        [0u64; crate::bench_timing::STAGE_COUNT]
                    };
                    *results_ptr.add(i) = (res, snap);
                }
            };

            if let Some(p) = pool.as_deref_mut() {
                p.parallel_for(n_chunk, run_task);
            } else {
                std::thread::scope(|scope| {
                    for i in 1..n_chunk {
                        scope.spawn(move || run_task(i));
                    }
                    run_task(0);
                });
            }

            for (_, snapshot) in &results {
                crate::bench_timing::merge_snapshot(snapshot);
            }
            for (res, _) in results {
                if let Err(e) = res {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
            if first_err.is_none() && self.collect_counts {
                for w in &workers {
                    self.counts.add_assign(&w.counts);
                }
            }
            for w in &mut workers {
                w.release_frame_refs();
            }
            parked = workers;
            if first_err.is_some() {
                break;
            }
            c0 = c1;
        }

        if let Some(pool_vec) = worker_pool {
            *pool_vec = parked;
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
