# Supported Features 

This page is used for tracking Cargo/Rust and CUDA features that are currently supported 
or planned to be supported in the future. As well as tracking some information about how they could 
be supported.

Note that `Not supported` does __not__ mean it won't ever be supported, it just means we haven't gotten
around to adding it yet.

## Opt-levels 

Fully supported and behaves mostly the same (because llvm is still used for optimizations). Except 
that libnvvm opts are run on anything except no-opts because nvvm only has -O0 and -O3

## Codegen units

Fully supported

## LTO

Not supported, but will be in the future (the code should mostly be the same from cg_llvm).

## Printing from kernels

Fully supported

## Allocating GPU memory from kernels

Fully supported

## libdevice math intrinsics 

Fully supported

## Thread/Block index and dimensions 

Fully supported

## Unified memory

Fully supported 

## CUDA Graphs

Partially supported

## Kernel assertions 

Fully supported 

## Kernel panics

Fully supported, however, it currently just traps instead of printing due to
mysterious CUDA errors. 

## Dynamic Parallelism

Not supported but shouldn't be much work.

This [sample](https://github.com/nvidia-compiler-sdk/nvvmir-samples/blob/master/device-side-launch/Device-Side-Launch.txt) States
it should be as simple as declaring the cuda_device_runtime_api.h functions (declared as extern functions) and wrapping them in cuda_std.

## Inline PTX asm

Not supported, most of the code can probably be taken from cg_llvm.

## Textures and Surfaces

Not supported. 

This is probably the hardest feature here, the exact details of how things like texture objects should be codegenned are
unclear. From limited tests with nvcc, it seems we can probably just pass them as u64 handles and do some conversions to 
get them to compile.

## Warp sync/reduce/vote functions

Not supported but shouldn't be much work.

## Warp matrix functions

Not supported.

## Shared memory

Not supported. Codegenning shared memory appears to be very involved, since it seems to be done
by declaring a lot of globals. We can probably replicate this by making `__nvvm_get_shared_mem_ptr(i64)`
intrinsics that the codegen recognizes calls to and codegens a global appropriately.
We also need inline assembly to read the `%dynamic_smem_size` PTX register to safety-check dynamic
shared mem. 

Dynamic shared mem also causes another problem because we must only allow shared mem to be queried
once. This is because dynamic shared mem if queried multiple times, just yields the same buffer. 
Which causes issues of data races and aliasing, so we must be very careful. We
can probably enforce this by enforcing that a call to `__nvvm_get_dynamic_shared_mem_ptr` can only:
- occur inside of a kernel function.
- occur only once inside of a kernel function.

These restrictions should not cause problems, if a kernel wants to give dynamic shared mem access to a child
function it can just pass a reference to it. It should also prevent a lot of confusing bugs
stemming from dynamic shared mem returning the same buffer.

The inherent unsafe nature of shared mem requires some bikeshedding over the API however.

## Atomic functions 

Not supported but shouldn't be much work.

## SIMD instructions 

Not supported, needs some investigation, it seems nowadays nvcc implements them as compiler builtins
and not as PTX instructions, but we have no idea what nvvm ir nvcc actually generates for them.

## PTX compiling APIs

Fully supported

## PTX Linking APIs

Fully supported

## Graphics API Interop

Not supported, needs some collaboration with wgpu developers.

## OptiX 

Not supported but will be in the future.

Host-side OptiX APIs are simple, just a standard C API wrapper like cust.

Device-side OptiX is a little bit more exotic. It seems that device-side OptiX works
by using functions which are implicitly defined by CUDA when used:

(OptiX SDK/include/internal/optix_7_device_impl.h)

```c
static __forceinline__ __device__ void optixSetPayload_0( unsigned int p )
{
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 0 ), "r"( p ) : );
}
```

This should be easy to emulate by just declaring extern functions which match the signatures 
in the header file. Most optix functions map to internal functions, but some are actual code
which can just be reimplemented (although we probably dont even need it because many are
linear algebra functions any crate can provide).

| Indicator | Meaning |
| --------- | ------- |
| ➖ | Not Applicable |
| ❌ | Not Supported |
| ✔️ | Fully Supported |
| 🟨 | Partially Supported |

# CUDA C++ Language Extensions

Note: Most of these categories are used __very__ rarely in CUDA code, therefore
do not be alarmed that it seems like many things are not supported. We just focus
on things used by the wide majority of users.

| Feature Name | Support Level | Notes |
| ------------ | ------------- | ----- |
| Function Execution Space Specifiers | ➖ |
| Variable Memory Space Specifiers | ➖ | Handled Implicitly |
| Built-in Vector Types | ➖ | Use linear algebra libraries like vek |
| Built-in Variables | ✔️ |
| Memory Fence Instructions | ✔️ |
| Synchronization Functions | ✔️ |
| Mathematical Functions | ✔️ |
| Texture Functions | ❌ |
| Surface Functions | ❌ |
| Read-Only Data Cache Load Function | ❌ | No real need, immutable references hint this automatically |
| Load Functions Using Cache Hints | ❌ |
| Store Functions Using Cache Hints | ❌ |
| Time Function | ✔️ | 
| Atomic Functions | ❌ |
| Address Space Predicate Functions | ➖ | Address Spaces are implicitly handled, but they may be added for exotic interop with CUDA C/C++ |
| Address Space Conversion Functions | ➖ |
| Alloca Function | ➖ |
| Compiler Optimization Hint Functions | ➖ | Existing `core` hints work |
| Warp Vote Functions | ❌ |
| Warp Match Functions | ❌ |
| Warp Reduce Functions | ❌ |
| Warp Shuffle Functions | ❌ |
| Nanosleep | ✔️ |
| Warp Matrix Functions | ❌ |
| Asynchronous Barrier | ❌ |
| Asynchronous Data Copies | ❌ |
| Profiler Counter Function | ✔️ |
| Assertion | ✔️ |
| Trap Function | ✔️ |
| Breakpoint | ✔️ |
| Formatted Output | ✔️ |
| Dynamic Global Memory Allocation | ✔️ |
| Execution Configuration | ✔️ |
| Launch Bounds | ✔️ |
| Pragma Unroll | ❌ |
| SIMD Video Instructions | ❌ |
| Cooperative Groups | ❌ |
| Dynamic Parallelism | ❌ |
| Stream Ordered Memory | ✔️ |
| Graph Memory Nodes | ❌ | Not supported, but Kernel launch nodes are |
| Unified Memory | ✔️ |
