//! Utilities for configuring code based on the specified compute capability.

use cuda_std_macros::gpu_only;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputeCapability {
    Compute35,
    Compute37,
    Compute50,
    Compute52,
    Compute53,
    Compute60,
    Compute61,
    Compute62,
    Compute70,
    Compute72,
    Compute75,
    Compute80,
    Compute86,
    Compute87,
    Compute89,
    Compute90,
    Compute100,
    Compute120,
}

impl ComputeCapability {
    /// Parses a compute capability from the `CUDA_ARCH` environment variable set by `cuda_builder`.
    /// This is a compile-time variable so any comparisons of the compute capability should expand to constant
    /// values.
    ///
    /// This allows you to use the current capability to decide what path to take in a function with the incorrect
    /// path being optimized away.
    #[gpu_only]
    #[inline(always)]
    pub fn from_cuda_arch_env() -> Self {
        // set by cuda_builder
        match env!("CUDA_ARCH") {
            "350" => ComputeCapability::Compute35,
            "370" => ComputeCapability::Compute37,
            "500" => ComputeCapability::Compute50,
            "520" => ComputeCapability::Compute52,
            "530" => ComputeCapability::Compute53,
            "600" => ComputeCapability::Compute60,
            "610" => ComputeCapability::Compute61,
            "620" => ComputeCapability::Compute62,
            "700" => ComputeCapability::Compute70,
            "720" => ComputeCapability::Compute72,
            "750" => ComputeCapability::Compute75,
            "800" => ComputeCapability::Compute80,
            "860" => ComputeCapability::Compute86,  // Ampere (RTX 30 series, A100)
            "870" => ComputeCapability::Compute87,  // Ampere (Jetson AGX Orin)
            "890" => ComputeCapability::Compute89,  // Ada Lovelace (RTX 40 series)
            "900" => ComputeCapability::Compute90,  // Hopper (H100)
            "1000" => ComputeCapability::Compute100, // Blackwell (RTX 50 series, H200, B100, CUDA 12.6 and later)
            "1200" => ComputeCapability::Compute120, // Blackwell (RTX 50 series, H200, B100, CUDA 12.8 and later)
            _ => panic!("CUDA_ARCH had an invalid value"),
        }
    }
}
