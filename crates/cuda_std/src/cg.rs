//! Cuda Cooperative Groups API interface.

use crate::gpu_only;

mod ffi {
    use core::ffi::c_void;

    pub type GridGroup = *mut c_void;
    extern "C" {
        pub(super) fn this_grid() -> GridGroup;
        pub(super) fn GridGroup_destroy(gg: GridGroup);
        pub(super) fn GridGroup_is_valid(gg: GridGroup) -> bool;
        pub(super) fn GridGroup_sync(gg: GridGroup);
        pub(super) fn GridGroup_size(gg: GridGroup) -> u64;
        pub(super) fn GridGroup_thread_rank(gg: GridGroup) -> u64;
        pub(super) fn GridGroup_num_threads(gg: GridGroup) -> u64;
        pub(super) fn GridGroup_num_blocks(gg: GridGroup) -> u64;
        pub(super) fn GridGroup_block_rank(gg: GridGroup) -> u64;
        // dim3 GridGroup_group_dim(); // TODO: impl these.
        // dim3 GridGroup_dim_blocks(); // TODO: impl these.
        // dim3 GridGroup_block_index(); // TODO: impl these.
    }
}

pub struct GridGroup(ffi::GridGroup);

impl Drop for GridGroup {
    fn drop(&mut self) {
        unsafe { ffi::GridGroup_destroy(self.0) }
    }
}

impl GridGroup {
    #[gpu_only]
    pub fn this_grid() -> Self {
        let ptr = unsafe { ffi::this_grid() };
        GridGroup(ptr)
    }

    #[gpu_only]
    pub fn is_valid(&mut self) -> bool {
        unsafe { ffi::GridGroup_is_valid(self.0) }
    }

    #[gpu_only]
    pub fn sync(&mut self) {
        unsafe { ffi::GridGroup_sync(self.0) }
    }

    #[gpu_only]
    pub fn size(&mut self) -> u64 {
        unsafe { ffi::GridGroup_size(self.0) }
    }

    #[gpu_only]
    pub fn thread_rank(&mut self) -> u64 {
        unsafe { ffi::GridGroup_thread_rank(self.0) }
    }

    #[gpu_only]
    pub fn num_threads(&mut self) -> u64 {
        unsafe { ffi::GridGroup_num_threads(self.0) }
    }

    #[gpu_only]
    pub fn num_blocks(&mut self) -> u64 {
        unsafe { ffi::GridGroup_num_blocks(self.0) }
    }

    #[gpu_only]
    pub fn block_rank(&mut self) -> u64 {
        unsafe { ffi::GridGroup_block_rank(self.0) }
    }
}
