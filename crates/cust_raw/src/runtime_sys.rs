//! Bindings to the CUDA Runtime API
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub use crate::types::driver::*;
pub use crate::types::surface::*;
pub use crate::types::texture::*;
pub use crate::types::vector::dim3;

include!(concat!(env!("OUT_DIR"), "/runtime_sys.rs"));
