//! Raw bindings to the OptiX 7.3 SDK.

#![allow(warnings)]

use cust_raw::*;
include!("../optix.rs");

extern "C" {
    pub fn optixInit() -> OptixResult;
}
