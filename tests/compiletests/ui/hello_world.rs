// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn add_one(x: *mut f32) {
    *x = *x + 1.0;
}
