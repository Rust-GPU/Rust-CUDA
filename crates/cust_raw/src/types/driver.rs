//! Bindings to driver types in the CUDA runtime API.
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]
use crate::types::vector::dim3;
include!(concat!(env!("OUT_DIR"), "/driver_types_sys.rs"));
