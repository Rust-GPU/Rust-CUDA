// Test CUDA thread synchronization functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::thread;

#[kernel]
pub unsafe fn test_synchronization() {
    // Test thread synchronization
    thread::sync_threads();

    // Test memory fences
    thread::device_fence();
    thread::grid_fence();
    thread::system_fence();
}
