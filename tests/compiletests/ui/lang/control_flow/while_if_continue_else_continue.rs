// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_while_if_continue_else_continue(i: i32) {
    while i < 10 {
        if i == 0 {
            continue;
        } else {
            continue;
        }
    }
}
