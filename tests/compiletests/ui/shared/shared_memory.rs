// Test CUDA shared memory allocations compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::{shared_array, thread};

#[kernel]
pub unsafe fn test_static_shared_memory() {
    // Allocate static shared memory for 256 i32 values
    let shared_data = shared_array![i32; 256];

    let tid = thread::thread_idx_x() as usize;

    // Write to shared memory
    *shared_data.add(tid) = tid as i32;

    // Synchronize threads before reading
    thread::sync_threads();

    // Read from shared memory
    let _value = *shared_data.add(tid);
}

#[kernel]
pub unsafe fn test_different_types() {
    // Test different array types
    let _shared_u32 = shared_array![u32; 128];
    let _shared_f32 = shared_array![f32; 64];
    let _shared_u8 = shared_array![u8; 512];

    // Test arrays of arrays
    let _shared_vec3 = shared_array![[f32; 3]; 32];
}
