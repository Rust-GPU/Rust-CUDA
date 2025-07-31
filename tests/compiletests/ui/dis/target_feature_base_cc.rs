// build-pass
// only-compute_70
// compile-flags: -Cllvm-args=--disassemble-entry=test_base_cc_70 --error-format=human

// This test verifies feature inheritance for compute_70 (base capability)

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_base_cc_70(result: *mut f32) {
    let mut val = 0.0f32;

    // arch=compute_70 should have target_feature=compute_70 enabled
    #[cfg(target_feature = "compute_70")]
    {
        val += 70.0;
    }

    // arch=compute_70 should also have target_feature=compute_60 enabled (lower capability)
    #[cfg(target_feature = "compute_60")]
    {
        val += 60.0;
    }

    // arch=compute_70 should NOT have target_feature=compute_80 enabled (higher capability)
    #[cfg(target_feature = "compute_80")]
    {
        val += 80.0;
    }

    // arch=compute_70 should NOT have target_feature=compute_90 enabled
    #[cfg(target_feature = "compute_90")]
    {
        val += 90.0;
    }

    // Prevent DCE - expected value should be 130.0 (70 + 60)
    core::ptr::write_volatile(result, val);
}
