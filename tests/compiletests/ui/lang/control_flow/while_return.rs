// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_while_return(i: i32) {
    while i < 10 {
        return;
    }
}
