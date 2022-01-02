# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

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

- Added `cust::memory::LockedBox`, same as `LockedBuffer` except for single elements.
- Added `cust::memory::cuda_malloc_async`.
- Added `cust::memory::cuda_free_async`.
- Added `impl AsyncCopyDestination<LockedBox<T>> for DeviceBox<T>` for async HtoD memcpy.
- Added the `bytemuck` feature which is enabled by default.
- `zeroed` functions on `DeviceBox` and others are no longer unsafe and instead now require `T: Zeroable`. The functions are only available with the `bytemuck` feature.
- Added `zeroed_async` to `DeviceBox`.
- Added `drop_async` to `DeviceBox`.
- Added `new_async` to `DeviceBox`.

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
