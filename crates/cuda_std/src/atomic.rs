//! Atomic Types for modification of numbers in multiple threads in a sound way.
//!
//! # Core Interop
//!
//! Every type in this module works on the CPU (targets outside of nvptx). However, [`core::sync::atomic`] types
//! do **NOT** work on the GPU currently. This is because CUDA atomics have some fundamental differences
//! that make representing them fully with existing core types impossible:
//!
//! - CUDA has block-scoped, device-scoped, and system-scoped atomics, core does not make such a distinction (obviously).
//! - CUDA trivially supports relaxed/acquire/release orderings on most architectures, but SeqCst and other orderings use
//! specialized instructions on compute capabilities 7.x+, but can be emulated with fences/membars on 7.x >. This makes it difficult
//! to hide away such details in the codegen.
//! - CUDA has hardware atomic floats, core does not.
//! - CUDA makes the distinction between "fetch, do operation, read" (`atom`) and "do operation" (`red`).
//! - Core thinks CUDA supports 8 and 16 bit atomics, this is a bug in the nvptx target but it is nevertheless an annoying detail
//! to silently trap on.
//!
//! Therefore we chose to go with the approach of implementing all atomics inside cuda_std. In the future, we may support
//! a subset of core atomics, but for now, you will have to use cuda_std atomics.

pub mod intrinsics;
