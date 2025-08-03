// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_for_range_signed(i: i32) {
    for _ in 0..i {}
}
