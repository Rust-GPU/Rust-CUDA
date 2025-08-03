// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_while_while_break(i: i32) {
    while i < 20 {
        while i < 10 {
            break;
        }
    }
}
