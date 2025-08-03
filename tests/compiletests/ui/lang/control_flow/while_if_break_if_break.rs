// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_while_if_break_if_break(i: i32) {
    while i < 10 {
        if i == 0 {
            break;
        }
        if i == 1 {
            break;
        }
    }
}
