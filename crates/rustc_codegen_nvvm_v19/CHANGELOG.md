# Changelog

Notable changes to this project will be documented in this file.

## Unreleased

- Added symbols for cuda_std to link to for warp intrinsics.
- Completely remove support for 32-bit CUDA (it was broken and it is essentially unused nowadays).
- Add a way to export the final llvm ir module fed to libnvvm.

## 0.2.3 - 1/2/22

- Fixed the `raw_eq` intrinsic being undefined.

## 0.2.2 - 12/5/21 

- Pass all ADTs directly, fixing certain structs being passed indirectly because they are scalar pairs.

## 0.2.1 - 12/5/21

- Update find_cuda_helper to 0.2

## 0.2.0 - 12/5/21

### Address Spaces 

CUDA Address Spaces have been mostly implemented. Statics that are not mut statics and do not rely on
interior mutability (are "freeze" types) are placed in constant memory (`__constant__` in CUDA C++), otherwise
they are placed in global memory (`__global__`). Currently this only happens for user-defined statics, not for
codegen-internal globals such as intermediate alloc globals.

An `#[address_space(...)]` macro has been added to cuda_std to complement this change. However, this macro
is mostly just for advanced users and internal support for things like shared memory. Improper use can 
cause undefined behavior, so its use is generally discouraged.

### Dead Code Elimination 

PTX files no longer include useless functions and globals, we have switched to an alternative
method of codegen for the final steps of the codegen. We no longer lazily-load modules using dependency graphs, 
we instead merge all the modules into one then run global DCE on it before giving it to libnvvm.

This means all of the dead code is gone before it gets to the libnvvm stage, drastically lowering the size of 
the built PTX and improving codegen performance. `cuda_std` also has a macro `#[externally_visible]` which can
be used if you want to keep a function around for things like linking multiple PTX files together.

### Libm override

The codegen now has the ability to override [`libm`](https://docs.rs/libm/latest/libm/) functions with 
[`libdevice`](https://docs.nvidia.com/cuda/libdevice-users-guide/introduction.html#introduction) intrinsics.

Libdevice is a bitcode library shipped with every CUDA SDK installation which provides float routines that
are optimized for the GPU and for specific GPU architectures. However, these routines are hard to use automatically because
no_std math crates typically use libm for float things. So users often ended up with needlessly slow or large PTX files
because they used "emulated" routines.

Now, by default (can be disabled in cuda_builder) the codegen will override libm functions with calls to libdevice automatically.
However, if you rely on libm for determinism, you must disable the overriding, since libdevice is not strictly deterministic.
This also makes PTX much smaller generally, in our example path tracer, it slimmed the PTX file from about `3800` LoC to `2300` LoC.

- Trace-level debug is compiled out for release now, decreasing the size of the codegen dll and improving compile times.
- Updated to nightly 12/4/21

## 0.1.1 - 11/26/21

- Fix things using the `bswap` intrinsic panicking.
- (internal) Run clippy and clean things up a bit.
