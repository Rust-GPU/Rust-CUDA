//! gpu_rand is the Rust CUDA Project's equivalent of cuRAND. cuRAND unfortunately does not work with
//! the CUDA Driver API, therefore, we reimplement (and extend) some of its algorithms and provide them in this crate.
//!
//! This crate is meant to be gpu-centric, which means it may special-case certain things to run faster on the GPU by using PTX
//! assembly. However, it is supposed to also work on the CPU, allowing you to reuse the same random states across CPU and GPU.
//!
//! A lot of the initial code is taken from the [rust-random project](https://github.com/rust-random) and modified to make it able to
//! pass to the GPU, as well as cleaning up certain things and updating it to edition 2021.
//! The following generators are implemented:
//!

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![allow(clippy::unreadable_literal)]
#![cfg_attr(target_os = "cuda", no_std)]
#![feature(doc_cfg)]

pub mod xoroshiro;

mod default;
mod gpurng;

pub use default::*;
pub use gpurng::*;
