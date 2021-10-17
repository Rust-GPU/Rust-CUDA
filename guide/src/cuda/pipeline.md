# The CUDA Pipeline

As you may already know, "traditional" cuda is usually in the form of CUDA C/C++ files which use `.cu` extension. These files 
can be compiled using NVCC (NVIDIA CUDA Compiler) into an executable.

CUDA files consist of **device** and **host** functions. **device** functions are functions that run on the GPU, also called kernels.
**host** functions run on the CPU and usually include logic on how to allocate GPU memory and call device functions.

However, a lot goes on behind the scenes that most people don't know about, a lot of it is integral to how rustc_codegen_nvvm works
so we will briefly go over it.

# Stages

The NVIDIA CUDA Compiler consists of distinct stages of compilation:

[![NVCC]](graphics/cuda-compilation-from-cu-to-executable.png)

NVCC separates device and host functions and compiles them separately. 
Most importantly, device functions are compiled to LLVM IR, and then the LLVM IR is fed to a library
called `libnvvm`.

`libnvvm` is a closed source library which takes in a subset of LLVM IR, it optimizes it further, then it
turns it into the next and most important stage of compilation, the PTX ISA.

PTX is a low level, assembly-like format with an open specification which can be targeted by any language.

We won't dig deep into what happens after PTX, but in essence, it is turned into a final format called SASS
which is register allocated and is finally sent to the GPU to execute.

# libnvvm

The stage/library we are most interested in is `libnvvm`. libnvvm is a closed source library that is 
distributed in every download of the CUDA SDK. Libnvvm takes a format called NVVM IR, it optimizes it, and 
converts it to a single PTX file you can run on NVIDIA GPUs using the driver or runtime API.

NVVM IR is a subset of LLVM IR, that is to say, it is a version of LLVM IR with restrictions. A couple 
of examples being:
- Many intrinsics are unsupported
- "Irregular" integer types such as `i4` or `i111` are unsupported and will segfault (however in theory they should be supported)
- Global names cannot include `.`.
- Some linkage types are not supported.
- Function ABIs are ignored, everything uses the PTX calling convention.

You can find the full specification of the NVVM IR [here](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html) if you are interested.

# Special PTX features

As far as an assembly format goes, PTX is fairly user friendly for a couple of reasons:
- It is well formatted.
- It is mostly fully specified (other than the iffy grammar specification).
- It uses named registers/parameters
- It uses virtual registers (since gpus have thousands of registers, listing all of them out would be unrealistic).
- It uses ASCII as a file encoding.
