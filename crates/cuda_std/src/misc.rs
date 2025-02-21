//! Misc functions that do not exactly fit into other categories.

use crate::gpu_only;
use core::arch::asm;

/// Suspends execution of the kernel, usually to pause at a specific point when debugging in a debugger.
#[gpu_only]
#[inline(always)]
pub fn breakpoint() {
    unsafe {
        asm!("brkpt");
    }
}

/// Increments a hardware counter between `0` and `7` (inclusive).
/// This function will increment the counter by one per warp.
///
/// # Panics
///
/// Panics if `counter` is not in the range of `0..=7`.
#[gpu_only]
#[inline(always)]
pub fn profiler_counter(counter: u32) {
    assert!(
        (0..=7).contains(&counter),
        "Profiler counter value must be in the range of 0..=7"
    );
    unsafe {
        asm!(
            "pmevent {}",
            in(reg32) counter
        )
    }
}

/// Returns the value of a per-multiprocessor counter incremented on every clock cycle.
#[gpu_only]
#[inline(always)]
pub fn clock() -> u64 {
    let mut clock;
    unsafe {
        asm!(
            "mov.u64 {}, %clock64",
            out(reg64) clock
        )
    }
    clock
}
