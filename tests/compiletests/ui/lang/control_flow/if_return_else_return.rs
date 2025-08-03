// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_if_return_else_return(i: i32) {
    if i < 10 {
        return;
    } else {
        return;
    }
}
