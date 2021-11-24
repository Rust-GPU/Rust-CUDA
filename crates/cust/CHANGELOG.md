# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

- Rename `DBuffer` -> `DeviceBuffer`. This is how it was in rustacuda, but it was changed
at some point, but now we reconsidered that it may be the wrong choice.
- Renamed `DBox` -> `DeviceBox`.
- Fixed some doctests that were using old APIs.
- Remove `GpuBox::as_device_ptr_mut` and `GpuBuffer::as_device_ptr_mut`.
- Change `GpuBox::as_device_ptr` and `GpuBuffer::as_device_ptr` to take `&self` instead of `&mut self`.
