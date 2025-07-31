// only-compute_101f
// build-fail
// compile-flags: -Cllvm-args=--disassemble-entry=test_family_cc_101f --error-format=human

// This test verifies feature inheritance for compute_101f (family capability)
// FIXME: This currently fails because NVVM doesn't support family suffixes like compute_101f
// This test is ignored until we use a later NVVM that supports family suffixes

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_family_cc_101f(result: *mut f32) {
    let mut val = 0.0f32;

    // compute_101f should have compute_101 enabled
    #[cfg(target_feature = "compute_101")]
    {
        val += 101.0;
    }

    // compute_101f should have compute_100 enabled (lower family version)
    #[cfg(target_feature = "compute_100")]
    {
        val += 100.0;
    }

    // compute_101f should NOT have compute_100f enabled (same family, lower minor)
    #[cfg(target_feature = "compute_100f")]
    {
        val += 100.5;
    }

    // compute_101f should also have lower capabilities enabled
    #[cfg(target_feature = "compute_90")]
    {
        val += 90.0;
    }

    #[cfg(target_feature = "compute_80")]
    {
        val += 80.0;
    }

    #[cfg(target_feature = "compute_70")]
    {
        val += 70.0;
    }

    // compute_101f should NOT have architecture-specific features
    #[cfg(target_feature = "compute_120a")]
    {
        val += 120.0;
    }

    // Prevent DCE - expected value should be 441.0 (101 + 100 + 90 + 80 + 70)
    core::ptr::write_volatile(result, val);
}
