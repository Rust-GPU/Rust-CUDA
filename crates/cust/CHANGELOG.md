# Changelog

Notable changes to this project will be documented in this file.

## Unreleased

- Add `memory::memcpy_dtoh` to allow copying from device to host.
- `DeviceSlice` is represented as a slice again, but as `[()]` instead of `[T]`.
- Reimplemented `Index` and `IndexMut` for `DeviceSlice` and removed `DeviceSlice::index`.
- Methods that returned `DeviceSlice` by value now return references:
    - `DeviceSlice::from_raw_parts`
    - `DeviceSlice::from_raw_parts_mut`
    - `DeviceSliceIndex::index`
    - `DeviceSliceIndex::get_unchecked`
- Added `DeviceSliceIndex::index_mut` and `DeviceSliceIndex::get_unchecked_mut`.
- Add support in `memory` for pitched malloc and 2D memcpy between device and host.
- `Stream::add_callback` now internally uses `cuStreamAddCallback` again, since there are no current plans to remove it (https://stackoverflow.com/a/58173486). As a result, the function again takes a device status as a parameter and *does* execute on context error.

## 0.3.2 - 2/16/22

- Fixed `ModuleJitOptions` not working (due to cursed reasons).

## 0.3.1 - 2/11/22

No changes, updated to cust_derive 0.2 to fix a bug.

## 0.3.0 - 2/7/22

### TLDR

This release is gigantic, so here are the main things you need to worry about:

`Context::create_and_push(FLAGS, device)` -> `Context::new(device)`.  
`Module::from_str(PTX)` -> `Module::from_ptx(PTX, &[])`.

### Context handling overhaul

The way that contexts are handled in cust has been completely overhauled, it now
uses primary context handling instead of the normal driver API context APIs. This 
is aimed at future-proofing cust for libraries such as cuBLAS and cuFFT, as well as
overall simplifying the context handling APIs. This does mean that the API changed a bit:
- `create_and_push` is now `new` and it only takes a device, not a device and flags.
- `set_flags` is now used for setting context flags.
- `ContextStack`, `UnownedContext`, and other legacy APIs are gone.

The old context handling is fully present in `cust::context::legacy` for anyone who needs it for specific reasons. If you use `quick_init` you don't need to worry about
any breaking changes, the API is the same.

### `cust_core`

`DeviceCopy` has now been split into its own crate, `cust_core`. The crate is `#![no_std]`, which allows you to
pull in `cust_core` in GPU crates for deriving `DeviceCopy` without cfg shenanigans.

### Removed

- `DeviceBox::wrap`, use `DeviceBox::from_raw`.
- `DeviceSlice::as_ptr` and `DeviceSlice::as_mut_ptr`. Use `DeviceSlice::as_device_ptr` then `DevicePointer::as_(mut)_ptr`.
- `DeviceSlice::chunks` and consequently `DeviceChunks`.
- `DeviceSlice::chunks_mut` and consequently `DeviceChunksMut`.
- `DeviceSlice::from_slice` and `DeviceSlice::from_slice_mut` because it was unsound.
- `DevicePointer::as_raw_mut` (use `DevicePointer::as_mut_ptr`).
- `DevicePointer::wrap` (use `DevicePointer::from_raw`).
- `DeviceSlice` no longer implements `Index` and `IndexMut`, switching away from `[T]` made this impossible to implement.
Instead you can now use `DeviceSlice::index` which behaves the same.
- `vek` is no longer re-exported.

### Deprecated

- `Module::from_str`, use `Module::from_ptx` and pass `&[]` for options.
- `Module::load_from_string`, use `Module::from_ptx_cstr`.

### Added 

- `cust::memory::LockedBox`, same as `LockedBuffer` except for single elements.
- `cust::memory::cuda_malloc_async`.
- `cust::memory::cuda_free_async`.
- `impl AsyncCopyDestination<LockedBox<T>> for DeviceBox<T>` for async HtoD/DtoH memcpy.
- `DeviceBox::new_async`.
- `DeviceBox::drop_async`.
- `DeviceBox::zeroed_async`.
- `DeviceBox::uninitialized_async`.
- `DeviceBuffer::uninitialized_async`.
- `DeviceBuffer::drop_async`.
- `DeviceBuffer::zeroed`.
- `DeviceBuffer::zeroed_async`.
- `DeviceBuffer::cast`.
- `DeviceBuffer::try_cast`.
- `DeviceSlice::set_8` and `DeviceSlice::set_8_async`.
- `DeviceSlice::set_16` and `DeviceSlice::set_16_async`.
- `DeviceSlice::set_32` and `DeviceSlice::set_32_async`.
- `DeviceSlice::set_zero` and `DeviceSlice::set_zero_async`.
- the `bytemuck` feature which is enabled by default.
- mint integration behind `impl_mint`.
- half integration behind `impl_half`.
- glam integration behind `impl_glam`.
- experimental linux external memory import APIs through `cust::external::ExternalMemory`.
- `DeviceBuffer::as_slice`.
- `DeviceVariable`, a simple wrapper around `DeviceBox<T>` and `T` which allows easy management of a CPU and GPU version of a type.
- `DeviceMemory`, a trait describing any region of GPU memory that can be described with a pointer + a length.
- `memcpy_htod`, a wrapper around `cuMemcpyHtoD_v2`.
- `mem_get_info` to query the amount of free and total memory.
- `DevicePointer::as_ptr` and `DevicePointer::as_mut_ptr` for `*const T` and `*mut T`.
- `DevicePointer::from_raw` for `CUdeviceptr -> DevicePointer<T>` with a safe function.
- `DevicePointer::cast`.
- dependency on `cust_core` for `DeviceCopy`.
- `ModuleJitOption`, `JitFallback`, `JitTarget`, and `OptLevel` for specifying options when loading a module. Note that
`ModuleJitOption::MaxRegisters` does not seem to work currently, but NVIDIA is looking into it.
You can achieve the same goal by compiling the ptx to cubin using nvcc then loading that: `nvcc --cubin foo.ptx -maxrregcount=REGS`
- `Module::from_fatbin`.
- `Module::from_cubin`.
- `Module::from_ptx` and `Module::from_ptx_cstr`.
- `Stream`, `Module`, `Linker`, `Function`, `Event`, `UnifiedBox`, `ArrayObject`, `LockedBuffer`, `LockedBox`, `DeviceSlice`, `DeviceBuffer`, and `DeviceBox` all now impl `Send` and `Sync`, this makes
it much easier to write multigpu code. The CUDA API is fully thread-safe except for graph objects.

### Changed 

- `zeroed` functions on `DeviceBox` and others are no longer unsafe and instead now require `T: Zeroable`. The functions are only available with the `bytemuck` feature.
- `Stream::add_callback` now internally uses `cuLaunchHostFunc` anticipating the deprecation and removal of `cuStreamAddCallback` per the driver docs. This does however mean that the function no longer takes a device status as a parameter and does not execute on context error.
- `Linker::complete` now only returns the built cubin, and not the cubin and a duration.
- Features such as `vek` for implementing DeviceCopy are now `impl_cratename`, e.g. `impl_vek`, `impl_half`, etc.
- `DevicePointer::as_raw` now returns a `CUdeviceptr` instead of a `*const T`.
- `num-complex` integration is now behind `impl_num_complex`, not `num-complex`.
- `DeviceBox` now requires `T: DeviceCopy` (previously it didn't but almost all its methods did).
- `DeviceBox::from_raw` now takes a `CUdeviceptr` instead of a `*mut T`.
- `DeviceBox::as_device_ptr` now requires `&self` instead of `&mut self`.
- `DeviceBuffer` now requires `T: DeviceCopy`.
- `DeviceBuffer` is now `repr(C)` and is represented by a `DevicePointer<T>` and a `usize`.
- `DeviceSlice` now requires `T: DeviceCopy`.
- `DeviceSlice` is now represented as a `DevicePointer<T>` and a `usize` (and is repr(C)) instead of `[T]` which was definitely unsound.
- `DeviceSlice::as_ptr` and `DeviceSlice::as_ptr_mut` now both return a `DevicePointer<T>`.
- `DeviceSlice` is now `Clone` and `Copy`.
- `DevicePointer::as_raw` now returns a `CUdeviceptr`, not a `*const T` (use `DevicePointer::as_ptr`).
- Fixed typo in `CudaError`, `InvalidSouce` is now `InvalidSource`, no more invalid sauce 🍅🥣

## 0.2.2 - 12/5/21

- Update find_cuda_helper to 0.2

## 0.2.1 - 12/5/21

- Added `Device::as_raw`.
- Added `MemoryAdvise` for unified memory advising.
- Added `MemoryAdvise::prefetch_host` and `MemoryAdvise::prefetch_device` for telling CUDA to explicitly fetch unified memory somewhere.
- Added `MemoryAdvise::advise_read_mostly`.
- Added `MemoryAdvise::preferred_location` and `MemoryAdvise::unset_preferred_location`.
Note that advising APIs are only present on high end GPUs such as V100s.
- `StreamFlags::NON_BLOCKING` has been temporarily disabled because of [soundness concerns](https://github.com/Rust-GPU/Rust-CUDA/issues/15).

## 0.2.0 - 11/26/21

- Change `GpuBox::as_device_ptr` and `GpuBuffer::as_device_ptr` to take `&self` instead of `&mut self`.
- Rename `DBuffer` -> `DeviceBuffer`. This is how it was in rustacuda, but it was changed
at some point, but now we reconsidered that it may be the wrong choice.
- Renamed `DBox` -> `DeviceBox`.
- Renamed `DSlice` -> `DeviceSlice`.

- Remove `GpuBox::as_device_ptr_mut` and `GpuBuffer::as_device_ptr_mut`.
- Remove accidentally added `vek` default feature.
- `vek` feature now uses `default-features = false`, this also means `Rgb` and `Rgba` no longer implement `DeviceCopy`.

- Fixed some doctests that were using old APIs.
