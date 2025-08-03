// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_for_range(i: u32) {
    for _ in 0..i {}
}
