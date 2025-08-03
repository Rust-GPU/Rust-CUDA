// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_while(i: i32) {
    while i < 10 {}
}
