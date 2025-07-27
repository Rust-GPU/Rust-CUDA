# Compute Capability Gating

This section covers how to write code that adapts to different CUDA compute capabilities
using conditional compilation.

## What are Compute Capabilities?

CUDA GPUs have different "compute capabilities" that determine which features they
support. Each capability is identified by a version number like `3.5`, `5.0`, `6.1`,
`7.5`, etc. Higher numbers generally mean more features are available.

For example:

- Compute capability 5.0+ supports 64-bit integer min/max and bitwise atomic operations
- Compute capability 6.0+ supports double-precision (f64) atomic operations
- Compute capability 7.0+ supports tensor core operations

For comprehensive details, see [NVIDIA's CUDA documentation on GPU architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation).

## Virtual vs Real Architectures

In CUDA terminology:

- **Virtual architectures** (`compute_XX`) define the PTX instruction set and available
  features
- **Real architectures** (`sm_XX`) represent actual GPU hardware

rust-cuda works exclusively with virtual architectures since it only generates PTX. The
`NvvmArch::ComputeXX` enum values correspond to CUDA's virtual architectures.

## Using Target Features

When you build a CUDA kernel with `cuda_builder`, the architecture you choose (e.g.,
`NvvmArch::Compute61`) enables target features that you can use for conditional compilation.

These features follow the pattern `compute_XX` where XX is the capability number without
the decimal point. The enabled feature means "at least this capability", matching
NVIDIA's semantics.

### Example: Basic Usage

```rust
use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("kernels")
        .arch(NvvmArch::Compute61)  // Target compute capability 6.1+
        .build()
        .unwrap();
}
```

This enables only the `compute_61` target feature, meaning the code requires
at least compute capability 6.1.

For other targeting patterns (exact ranges, maximum capabilities), use boolean
`cfg` logic as shown in the examples below.

### Manual Compilation (Without CudaBuilder)

If you're invoking `rustc` directly instead of using `cuda_builder`, you only need to specify the architecture through LLVM args:

```bash
rustc --target nvptx64-nvidia-cuda \
    -C llvm-args=-arch=compute_61 \
    -Z codegen-backend=/path/to/librustc_codegen_nvvm.so \
    ...
```

Or with cargo:

```bash
export RUSTFLAGS="-C llvm-args=-arch=compute_61 -Z codegen-backend=/path/to/librustc_codegen_nvvm.so"
cargo build --target nvptx64-nvidia-cuda
```

The codegen backend automatically synthesizes all appropriate target features based on the architecture. For example, targeting `compute_61` will enable `compute_35`, `compute_37`, `compute_50`, `compute_52`, `compute_53`, `compute_60`, and `compute_61` features for conditional compilation.

## Conditional Compilation in Kernels

You can use `#[cfg(target_feature = "compute_XX")]` to conditionally compile code based on the available compute capabilities. With boolean logic, you can express any capability range you need.

### Common Patterns

#### At Least a Capability (Default)
```rust,no_run
// Code that requires compute 6.0 or higher
#[cfg(target_feature = "compute_60")]
{
    cuda_std::atomic::atomic_add(data, 1.0); // f64 atomics need 6.0+
}
```

#### Exactly One Capability  
```rust,no_run
// Code that targets exactly compute 5.0 (not 5.2+)
#[cfg(all(target_feature = "compute_50", not(target_feature = "compute_52")))]
{
    // Optimizations specific to compute 5.0
}

// Code that targets exactly compute 6.1 (not 6.2+)
#[cfg(all(target_feature = "compute_61", not(target_feature = "compute_62")))]
{
    // Features specific to compute 6.1
}
```

#### Up To a Maximum Capability
```rust,no_run
// Code that works on compute 5.0 and below (not 5.2+)
#[cfg(all(target_feature = "compute_35", not(target_feature = "compute_52")))]
{
    // Fallback implementation for older GPUs
}

// Code that works up to compute 6.0 (not 6.1+)  
#[cfg(all(target_feature = "compute_35", not(target_feature = "compute_61")))]
{
    // Maximum compatibility implementation
}
```

#### Capability Ranges
```rust,no_run
// Code that works on compute 5.0 through 7.0 (not 7.2+)
#[cfg(all(target_feature = "compute_50", not(target_feature = "compute_72")))]
{
    // Features available in this range
}
```

### Complete Example

```rust,no_run
use cuda_std::*;

#[kernel]
pub unsafe fn adaptive_kernel(data: *mut f64) {
    // This code only compiles when targeting compute 6.0 or higher
    #[cfg(target_feature = "compute_60")]
    {
        // f64 atomics are only available on compute 6.0+
        cuda_std::atomic::atomic_add(data, 1.0);
    }

    // Fallback for older GPUs
    #[cfg(not(target_feature = "compute_60"))]
    {
        // Manual implementation or alternative approach
    }
}
```

## Best Practices

### 1. Choose the Lowest Viable Architecture

Select the lowest compute capability that provides the features you need. This maximizes GPU compatibility:

```rust,no_run
// If you only need basic atomics
.arch(NvvmArch::Compute35)

// If you need 64-bit integer atomics
.arch(NvvmArch::Compute50)

// If you need f64 atomics
.arch(NvvmArch::Compute60)
```

### 2. Provide Fallbacks When Possible

For maximum compatibility, provide alternative implementations for older GPUs:

```rust,no_run
#[cfg(target_feature = "compute_50")]
fn fast_path(data: *mut u64) {
    // Use hardware atomic
    atomic_min(data, 100);
}

#[cfg(not(target_feature = "compute_50"))]
fn fast_path(data: *mut u64) {
    // Software fallback
}
```

## Debugging Capability Issues

If you encounter errors about missing functions or features:

1. Check the compute capability you're targeting in `cuda_builder`
2. Verify your GPU supports the features you're using
3. Use `nvidia-smi` to check your GPU's compute capability
4. Add appropriate `#[cfg]` guards or increase the target architecture

## Runtime Behavior

Again, rust-cuda **only generates PTX**, not pre-compiled GPU binaries
("[fatbinaries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fatbinaries)").
This PTX is then JIT-compiled by the CUDA driver at _runtime_.

For more details, see [NVIDIA's documentation on GPU
compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation)
and [JIT
compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#just-in-time-compilation).
