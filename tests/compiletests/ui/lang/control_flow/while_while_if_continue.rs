// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_while_while_if_continue(i: i32) {
    while i < 20 {
        while i < 10 {
            if i > 5 {
                continue;
            }
        }
    }
}
