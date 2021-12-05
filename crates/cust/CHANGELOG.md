# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

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
