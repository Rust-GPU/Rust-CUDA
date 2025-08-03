// Test CUDA thread indexing functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::thread;

#[kernel]
pub unsafe fn test_thread_indices() {
    // Thread indices within block
    let _tx = thread::thread_idx_x();
    let _ty = thread::thread_idx_y();
    let _tz = thread::thread_idx_z();

    // Block indices within grid
    let _bx = thread::block_idx_x();
    let _by = thread::block_idx_y();
    let _bz = thread::block_idx_z();

    // Block dimensions
    let _bdx = thread::block_dim_x();
    let _bdy = thread::block_dim_y();
    let _bdz = thread::block_dim_z();

    // Grid dimensions
    let _gdx = thread::grid_dim_x();
    let _gdy = thread::grid_dim_y();
    let _gdz = thread::grid_dim_z();
}
