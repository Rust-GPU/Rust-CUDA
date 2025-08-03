// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_if_return_else(i: i32) {
    if i < 10 {
        return;
    } else {
    }
}
