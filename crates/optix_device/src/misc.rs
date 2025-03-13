#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::gpu_only;

/// Retrieves the data past the SBT header for this particular program
///
/// # Safety
///
/// The type requested must match with what is stored in the SBT.
#[gpu_only]
pub unsafe fn sbt_data<T>() -> &'static T {
    let ptr: *const T;
    asm!("call ({}), _optix_get_sbt_data_ptr_64, ();", out(reg64) ptr);
    &*ptr
}
