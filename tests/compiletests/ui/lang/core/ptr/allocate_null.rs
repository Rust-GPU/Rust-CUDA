// Tests allocating a null pointer at `const` time.
// build-pass

use core::ptr::null;
use cuda_std::kernel;

const NULL_PTR: *const i32 = null();

#[kernel]
pub unsafe fn test_allocate_null() {
    let _null_ptr = NULL_PTR;
}
