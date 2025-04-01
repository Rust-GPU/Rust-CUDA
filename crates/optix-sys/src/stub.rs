use crate::optix_sys::OptixResult;

unsafe extern "C" {
    pub fn optixInit() -> OptixResult;
}
