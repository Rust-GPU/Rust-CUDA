// only-compute_120a
// build-fail
// compile-flags: -Cllvm-args=--disassemble-entry=test_arch_cc_120a --error-format=human

// This test verifies feature inheritance for compute_120a (architecture capability)
// FIXME: This currently fails because NVVM doesn't support architecture suffixes like compute_120a
// This test is ignored until we use a later NVVM that supports architecture suffixes

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_arch_cc_120a(result: *mut f32) {
    let mut val = 0.0f32;

    // compute_120a should have compute_120 enabled (base of architecture)
    #[cfg(target_feature = "compute_120")]
    {
        val += 120.0;
    }

    // compute_120a should also have all lower capabilities enabled
    #[cfg(target_feature = "compute_100")]
    {
        val += 100.0;
    }

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

    // compute_120a should NOT have family features from lower versions
    #[cfg(target_feature = "compute_100f")]
    {
        val += 100.5;
    }

    #[cfg(target_feature = "compute_101f")]
    {
        val += 101.5;
    }

    // Prevent DCE - expected value should be 460.0 (120 + 100 + 90 + 80 + 70)
    core::ptr::write_volatile(result, val);
}
