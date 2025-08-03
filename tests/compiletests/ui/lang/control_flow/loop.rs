// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_loop() {
    loop {}
}
