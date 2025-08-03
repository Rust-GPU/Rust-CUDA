// Tests using a vector like pointer at `const` time.
// build-pass

#![allow(internal_features)]
#![feature(ptr_internals)]

use core::ptr::Unique;
use cuda_std::kernel;

const VEC_LIKE: (Unique<usize>, usize, usize) = (Unique::<usize>::dangling(), 0, 0);

pub fn assign_vec_like() {
    let _vec_like = VEC_LIKE;
}

#[kernel]
pub unsafe fn test_allocate_vec_like() {
    assign_vec_like();
}
