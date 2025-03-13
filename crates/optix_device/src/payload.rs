use crate::sys;
#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::gpu_only;

/// Overrides the payload for the given register to a value.
///
/// # Panics
///
/// Panics if the index is higher than 7.
pub fn set_payload(idx: u8, p: u32) {
    // SAFETY: setting the payload cannot cause UB, only getting the value of an unset
    // register can.
    unsafe { sys::set_payload(idx, p) }
}

/// Returns the payload in the given register.
///
/// # Safety
///
/// The payload must have been set by the ray dispatch or by a program which came before this.
/// Moreover, the payload must have not been cleared with [`clear_register`].
///
/// # Panics
///
/// Panics if the index is higher than `7`.
pub unsafe fn get_payload(idx: u8) -> u32 {
    sys::get_payload(idx)
}

/// Clears the payload in the given register with an undefined value to ease register usage for programs
/// down the line.
///
/// The register is overriden with an undefined value and it must not be read until it is set again.
///
/// # Panics
///
/// Panics if the index is higher than `7`.
#[gpu_only]
pub fn clear_register(idx: u8) {
    assert!(idx <= 7, "Register index must be in range 0..=7");
    let idx = idx as u32;
    // its unclear whether optix_undef_value is just a compiler
    // hint for the optix jit compiler or if it returns a random
    // unspecified value. Uninit values are not allowed to exist
    // without MaybeUninit in rust, so just to be safe we do not
    // expose the value, and instead directly use inline asm to
    // create the value and then set it as a payload to avoid this.
    unsafe {
        asm!(
           "call ({tmp}), _optix_undef_value, ();",
           "call _optix_set_payload, ({}, {tmp});",
           in(reg32) idx,
           tmp = out(reg32) _,
        );
    }
}
