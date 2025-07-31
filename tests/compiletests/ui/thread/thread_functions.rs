// build-pass

// This test verifies CUDA thread functions are available and working

use cuda_std::kernel;
use cuda_std::thread;

#[kernel]
pub unsafe fn thread_functions_test() {
    // Thread identification functions
    let _tid = thread::thread_idx_x();
    let _bid = thread::block_idx_x();
    let _bdim = thread::block_dim_x();

    // Synchronization function
    thread::sync_threads();
}
