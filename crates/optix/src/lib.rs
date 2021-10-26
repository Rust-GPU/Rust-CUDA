pub mod context;
pub mod denoiser;
pub mod error;

pub use cust;
use error::{OptixResult, ToResult};
pub use optix_sys as sys;

/// Initializes the OptiX library. This must be called before using any OptiX function. It may
/// be called before or after initializing CUDA.
pub fn init() -> OptixResult<()> {
    // avoid initializing multiple times because that will try to load the dll every time.
    if !optix_is_initialized() {
        init_cold()
    } else {
        Ok(())
    }
}

#[cold]
#[inline(never)]
fn init_cold() -> OptixResult<()> {
    unsafe { sys::optixInit().to_result() }
}

/// Whether OptiX is initialized. If you are calling raw [`sys`] functions you must make sure
/// this is true, otherwise OptiX will segfault. In the safe wrapper it is done automatically and optix not
/// being initialized will return an error result.
pub fn optix_is_initialized() -> bool {
    // SAFETY: C globals are explicitly defined to be zero-initialized, and the sys version uses
    // Option for each field, and None is explicitly defined to be represented as a nullptr for Option<fn()>,
    // so its default should be the same as the zero-initialized global.
    // And, while we do not currently expose it, optix library unloading zero initializes the global.
    unsafe { g_optixFunctionTable != sys::OptixFunctionTable::default() }
}

extern "C" {
    pub(crate) static g_optixFunctionTable: sys::OptixFunctionTable;
}

/// Call a raw OptiX sys function, making sure that OptiX is initialized. Returning
/// an OptixNotInitialized error if it is not initialized. See [`optix_is_initialized`].
#[macro_export]
macro_rules! optix_call {
    ($name:ident($($param:expr),* $(,)?)) => {{
          if !$crate::optix_is_initialized() {
              Err($crate::error::OptixError::OptixNotInitialized)
          } else {
              <$crate::sys::OptixResult as $crate::error::ToResult>::to_result($crate::sys::$name($($param),*))
          }
    }};
}
