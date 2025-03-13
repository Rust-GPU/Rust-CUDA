//! Generic traits over raw FFI functions for floats, doubles, complex numbers, and double complex numbers.
//!
//! The functions are still very unsafe and do nothing except dispatch to the correct FFI function.

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

mod level1;
mod level3;

pub use level1::*;
pub use level3::*;
