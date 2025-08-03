// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_if_if(i: i32) {
    if i > 0 {
        if i < 10 {}
    }
}
