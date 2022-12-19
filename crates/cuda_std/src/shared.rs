//! Static and Dynamic shared memory handling.

use crate::gpu_only;

/// Statically allocates a buffer large enough for `len` elements of `array_type`, yielding
/// a `*mut array_type` that points to uninitialized shared memory. `len` must be a constant expression.
///
/// Note that this allocates the memory __statically__, it expands to a static in the `shared` address space.
/// Therefore, calling this macro multiple times in a loop will always yield the same data. However, separate
/// invocations of the macro will yield different buffers.
///
/// The data is uninitialized by default, therefore, you must be careful to not read the data before it is written to.
/// The semantics of what "uninitialized" actually means on the GPU (i.e. if it yields unknown data or if it is UB to read it whatsoever)
/// are not well known, so even if the type is valid for any backing memory, make sure to not read uninitialized data.
///
/// # Safety
///
/// Shared memory usage is fundamentally extremely unsafe and impossible to statically prove, therefore
/// the burden of correctness is on the user. Some of the things you must ensure in your usage of
/// shared memory are:
/// - Shared memory is only shared across __thread blocks__, not the entire device, therefore it is
/// unsound to try and rely on sharing data across more than one block.
/// - You must write to the shared buffer before reading from it as the data is uninitialized by default.
/// - [`thread::sync_threads`](crate::thread::sync_threads) must be called before relying on the results of other
/// threads, this ensures every thread has reached that point before going on. For example, reading another thread's
/// data after writing to the buffer.
/// - No access may be out of bounds, this usually means making sure the amount of threads and their dimensions are correct.
///
/// It is suggested to run your executable in `cuda-memcheck` to make sure usages of shared memory are right.
///
/// # Examples
///
/// ```no_run
/// #[kernel]
/// pub unsafe fn reverse_array(d: *mut i32, n: usize) {
///    let s = shared_array![i32; 64];
///    let t = thread::thread_idx_x() as usize;
///    let tr = n - t - 1;
///    *s.add(t) = *d.add(t);
///    thread::sync_threads();
///    *d.add(t) = *s.add(tr);
/// }
/// ```
#[macro_export]
macro_rules! shared_array {
    ($array_type:ty; $len:expr) => {{
        #[$crate::gpu_only]
        #[inline(always)]
        fn shared_array() -> *mut $array_type {
            use ::core::{cell::UnsafeCell, mem::MaybeUninit};
            struct SyncWrapper(UnsafeCell<MaybeUninit<[$array_type; $len]>>);
            // SAFETY: it is up to the user to verify sound shared memory usage, we cannot
            // fundamentally check it for soundness.
            unsafe impl Send for SyncWrapper {}
            // SAFETY: see above
            unsafe impl Sync for SyncWrapper {}

            // the initializer is discarded when declaring shared globals, so it is unimportant.
            #[$crate::address_space(shared)]
            static SHARED: SyncWrapper = SyncWrapper(UnsafeCell::new(MaybeUninit::uninit()));

            SHARED.0.get() as *mut $array_type
        }
        shared_array()
    }};
}

/// Gets a pointer to the dynamic shared memory that was allocated by the caller of the kernel. The
/// data is left uninitialized.
///
/// **Calling this function multiple times will yield the same pointer**.  
#[gpu_only]
pub fn dynamic_shared_mem<T>() -> *mut T {
    // it is unclear whether an alignment of 16 is actually required for correctness, however,
    // it seems like nvcc always generates the global with .align 16 no matter the type, so we just copy
    // nvcc's behavior for now.
    extern "C" {
        // need to use nvvm_internal and not address_space because address_space only parses
        // static definitions, not extern static definitions.
        #[rust_cuda::nvvm_internal(addrspace(3))]
        #[allow(improper_ctypes)]
        // mangle it a bit to make sure nobody makes the same thing
        #[link_name = "_Zcuda_std_dyn_shared"]
        static DYN_SHARED: ::core::cell::UnsafeCell<u128>;
    }

    // SAFETY: extern statics is how dynamic shared mem is done in CUDA. This will turn into
    // an extern variable decl in ptx, which is the same thing nvcc does if you dump the ptx from a cuda file.
    unsafe { DYN_SHARED.get() as *mut T }
}
