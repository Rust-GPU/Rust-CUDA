//! Bindings to CUDA texture types.
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
include!(concat!(env!("OUT_DIR"), "/texture_types_sys.rs"));
