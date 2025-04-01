#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub const LIBDEVICE_BITCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libdevice.bc"));

include!(concat!(env!("OUT_DIR"), "/nvvm_sys.rs"));
