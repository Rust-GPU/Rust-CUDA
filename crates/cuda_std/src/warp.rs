//! Functions that work over warps of threads.
//!
//! Warps in CUDA are groups of 32 threads that are dispatched together inside of
//! thread blocks and execute in SIMT fashion.

use crate::gpu_only;

/// Synchronizes all of the threads inside of this warp according to `mask`.
///
/// # Safety
///
/// The behavior of this function is undefined if:
/// - Any thread inside `mask` has exited.
/// - The executing thread is not inside of `mask`.
///
/// Moreover, on compute_62 and below, all of the threads inside `mask` must call
/// `sync` with the __exact same__ mask. Otherwise it is undefined behavior.
#[gpu_only]
#[inline(always)]
pub unsafe fn sync_warp(mask: u32) {
    extern "C" {
        #[link_name = "llvm.nvvm.bar.warp.sync"]
        fn sync(mask: u32);
    }

    sync(mask);
}

/// Returns the thread's lane within its warp. This value ranges from `0` to `WARP_SIZE - 1` (`WARP_SIZE` is 32 on all
/// architectures currently).
#[gpu_only]
#[inline(always)]
pub fn lane_id() -> u32 {
    let mut out;
    unsafe {
        asm!(
            "mov.u32 {}, %laneid",
            out(reg32) out
        );
    }
    out
}
