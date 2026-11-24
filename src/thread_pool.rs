//! Minimal, pure-`std` persistent thread pool owned by [`crate::Decoder`].
//!
//! Spawns worker threads lazily on first parallel call and joins them cleanly when dropped,
//! eliminating the per-frame OS thread creation/teardown overhead of `std::thread::scope`
//! (especially costly on Windows, where spawning 4 tile workers + 16 loop-filter threads per
//! frame costs milliseconds and triggers scheduler contention).

use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

#[inline]
fn lock_unpoisoned<T>(mu: &Mutex<T>) -> MutexGuard<'_, T> {
    mu.lock().unwrap_or_else(|e| e.into_inner())
}

#[inline]
fn wait_unpoisoned<'a, T>(cv: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(guard).unwrap_or_else(|e| e.into_inner())
}

/// RAII soundness guard ensuring [`ThreadPool::parallel_for`] always waits until every
/// dispatched worker finishes (`remaining == 0`) before returning or unwinding, so the
/// stack-local closure `&F` can never be dropped while a worker still holds a pointer to it.
struct DispatchGuard<'a> {
    shared: &'a PoolShared,
}

impl DispatchGuard<'_> {
    fn wait_all(&self) {
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) > 0 {
            if spins < 64 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                let mut guard = lock_unpoisoned(&self.shared.done_mu);
                while self.shared.remaining.load(Ordering::Acquire) > 0 {
                    guard = wait_unpoisoned(&self.shared.wake_caller, guard);
                }
                break;
            }
        }
    }
}

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        self.wait_all();
    }
}

struct DispatchState {
    /// Monotonically incremented on every `parallel_for` dispatch.
    generation: u64,
    /// Total task count for the active dispatch (`worker_id` runs iff `worker_id < active_tasks`).
    active_tasks: usize,
    /// Type-erased trampoline `|data, task_id| (*(data as *const F))(task_id)`.
    trampoline: fn(*const (), usize),
    /// Erased pointer to `&F` on the caller's stack; valid until `remaining == 0`.
    data: *const (),
    /// Set to `true` on [`ThreadPool::drop`] to wake and exit all workers.
    shutdown: bool,
}

// SAFETY: `data` points to an `F: Sync` value on the dispatching thread's stack, and
// `ThreadPool::parallel_for` blocks until `remaining == 0` (every worker finished executing
// `trampoline(data, id)`) before returning or unwinding.
unsafe impl Send for DispatchState {}

struct PoolShared {
    state: Mutex<DispatchState>,
    wake_workers: Condvar,
    /// Number of worker tasks still running in the active dispatch.
    remaining: AtomicUsize,
    /// Synchronizes caller wakeup (preventing lost notifications) and stores the first worker
    /// panic payload, if any.
    done_mu: Mutex<Option<Box<dyn Any + Send>>>,
    wake_caller: Condvar,
}

/// Persistent worker pool supporting synchronous scoped `parallel_for` execution.
///
/// Taking `&mut self` on [`Self::parallel_for`] statically guarantees at most one active
/// dispatch at a time, avoiding re-entrancy locks or heap-allocated job queues.
pub(crate) struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    shared: Arc<PoolShared>,
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("workers", &self.workers.len())
            .finish()
    }
}

fn noop_trampoline(_: *const (), _: usize) {}

impl ThreadPool {
    /// Creates an empty pool with zero spawned threads. Threads are created lazily on demand.
    pub(crate) fn new() -> Self {
        Self {
            workers: Vec::new(),
            shared: Arc::new(PoolShared {
                state: Mutex::new(DispatchState {
                    generation: 0,
                    active_tasks: 0,
                    trampoline: noop_trampoline,
                    data: std::ptr::null(),
                    shutdown: false,
                }),
                wake_workers: Condvar::new(),
                remaining: AtomicUsize::new(0),
                done_mu: Mutex::new(None),
                wake_caller: Condvar::new(),
            }),
        }
    }

    /// Executes `f(0), f(1), ..., f(n_tasks - 1)` in parallel: task `0` runs on the calling
    /// thread and tasks `1..n_tasks` run on persistent worker threads. Blocks until all tasks
    /// finish (including on panic, propagating the first panic once all workers settle).
    pub(crate) fn parallel_for<F>(&mut self, n_tasks: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        if n_tasks == 0 {
            return;
        }
        if n_tasks == 1 {
            f(0);
            return;
        }

        fn trampoline<F: Fn(usize) + Sync>(data: *const (), task_id: usize) {
            // SAFETY: caller ensures `data` points to a live `F` until `remaining == 0`.
            let f = unsafe { &*data.cast::<F>() };
            f(task_id);
        }

        let needed_workers = n_tasks - 1;
        if self.workers.len() < needed_workers {
            let initial_gen = lock_unpoisoned(&self.shared.state).generation;
            while self.workers.len() < needed_workers {
                let worker_id = self.workers.len() + 1;
                let shared = Arc::clone(&self.shared);
                let handle = thread::Builder::new()
                    .name(format!("vp9dec-worker-{worker_id}"))
                    .spawn(move || worker_loop(shared, worker_id, initial_gen))
                    .expect("failed to spawn vp9dec worker thread");
                self.workers.push(handle);
            }
        }

        // Publish the new generation under `state` lock before waking workers.
        {
            let mut state = lock_unpoisoned(&self.shared.state);
            state.active_tasks = n_tasks;
            state.trampoline = trampoline::<F>;
            state.data = (&f as *const F).cast::<()>();
            state.generation = state.generation.wrapping_add(1);
            self.shared
                .remaining
                .store(needed_workers, Ordering::Release);
        }
        // Guard is created immediately after publishing so even an unexpected unwind waits
        // until `remaining == 0` before `&f` leaves scope.
        let dispatch_guard = DispatchGuard {
            shared: &self.shared,
        };
        self.shared.wake_workers.notify_all();

        // Execute task 0 on the calling thread while workers execute `1..n_tasks`.
        let caller_res = catch_unwind(AssertUnwindSafe(|| f(0)));

        // Wait for every worker to finish before `&f` leaves scope (soundness invariant).
        drop(dispatch_guard);

        let worker_panic = lock_unpoisoned(&self.shared.done_mu).take();

        if let Err(payload) = caller_res {
            resume_unwind(payload);
        }
        if let Some(payload) = worker_panic {
            resume_unwind(payload);
        }
    }
}

fn worker_loop(shared: Arc<PoolShared>, worker_id: usize, mut seen_gen: u64) {
    loop {
        let (trampoline, data) = {
            let mut state = lock_unpoisoned(&shared.state);
            loop {
                if state.shutdown {
                    return;
                }
                if state.generation != seen_gen {
                    seen_gen = state.generation;
                    if worker_id < state.active_tasks {
                        break (state.trampoline, state.data);
                    }
                }
                state = wait_unpoisoned(&shared.wake_workers, state);
            }
        };

        let res = catch_unwind(AssertUnwindSafe(|| trampoline(data, worker_id)));
        if let Err(payload) = res {
            let mut guard = lock_unpoisoned(&shared.done_mu);
            if guard.is_none() {
                *guard = Some(payload);
            }
        }

        if shared.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Acquire and release `done_mu` before notifying so the caller cannot observe
            // `remaining > 0` and then miss this notification.
            let _guard = lock_unpoisoned(&shared.done_mu);
            drop(_guard);
            shared.wake_caller.notify_one();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if self.workers.is_empty() {
            return;
        }
        {
            let mut state = lock_unpoisoned(&self.shared.state);
            state.shutdown = true;
        }
        self.shared.wake_workers.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ThreadPool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_runs_varying_task_counts_and_reuses_workers() {
        let mut pool = ThreadPool::new();
        for &n in &[0usize, 1, 4, 2, 8, 1, 3, 6, 4] {
            let slots: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.parallel_for(n, |i| {
                slots[i].fetch_add(i + 1, Ordering::Relaxed);
            });
            for (i, slot) in slots.iter().enumerate() {
                assert_eq!(slot.load(Ordering::Relaxed), i + 1);
            }
        }
    }

    #[test]
    fn parallel_for_propagates_worker_panic_and_stays_usable() {
        let mut pool = ThreadPool::new();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.parallel_for(4, |i| {
                if i == 2 {
                    panic!("worker boom");
                }
            });
        }));
        assert!(res.is_err());

        // Pool remains sound and usable after a caught panic.
        let sum = AtomicUsize::new(0);
        pool.parallel_for(4, |i| {
            sum.fetch_add(i + 1, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn parallel_for_waits_for_workers_when_caller_task_panics() {
        let mut pool = ThreadPool::new();
        let finished = AtomicUsize::new(0);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.parallel_for(4, |i| {
                if i == 0 {
                    panic!("caller boom");
                }
                finished.fetch_add(1, Ordering::Relaxed);
            });
        }));
        assert!(res.is_err());
        // All 3 worker tasks must have completed before `parallel_for` unwound.
        assert_eq!(finished.load(Ordering::Relaxed), 3);

        // Pool remains usable after caller-side panic.
        let sum = AtomicUsize::new(0);
        pool.parallel_for(4, |i| {
            sum.fetch_add(i + 1, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }
}
