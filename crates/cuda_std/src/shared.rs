//! Shared memory handling. Currently only macros.

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
        // the initializer is discarded when declaring shared globals, so it is unimportant.
        #[$crate::address_space(shared)]
        static mut SHARED: MaybeUninit<[$array_type; $len]> = MaybeUninit::uninit();
        SHARED.as_mut_ptr() as *mut $array_type
    }};
}
