// Test creating uninitialized memory.
// build-pass

use core::mem::MaybeUninit;
use cuda_std::kernel;

const MAYBEI32: MaybeUninit<&i32> = MaybeUninit::<&i32>::uninit();

pub fn create_uninit_and_write() {
    let mut maybei32 = MAYBEI32;
    unsafe {
        maybei32.as_mut_ptr().write(&0);
    }
    let _maybei32 = unsafe { maybei32.assume_init() };
}

#[kernel]
pub unsafe fn test_uninit_memory() {
    create_uninit_and_write();
}
