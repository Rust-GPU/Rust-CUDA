// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_if_while(i: i32) {
    if i == 0 {
        while i < 10 {}
    }
}
