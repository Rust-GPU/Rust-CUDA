// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_if_else_if_else(i: i32) {
    if i > 0 {
    } else if i < 0 {
    } else {
    }
}
