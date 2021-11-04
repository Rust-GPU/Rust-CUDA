pub mod acceleration;
pub mod context;
pub mod curve_array;
pub mod custom_primitive_array;
pub mod denoiser;
pub mod error;
pub mod instance_array;
pub mod module;
pub mod pipeline;
pub mod program_group;
pub mod shader_binding_table;
pub mod sys;
pub mod triangle_array;

pub use cust;
use error::{Error, ToResult};
type Result<T, E = Error> = std::result::Result<T, E>;

/// Initializes the OptiX library. This must be called before using any OptiX function. It may
/// be called before or after initializing CUDA.
pub fn init() -> Result<()> {
    // avoid initializing multiple times because that will try to load the dll every time.
    if !optix_is_initialized() {
        init_cold()
    } else {
        Ok(())
    }
}

#[cold]
#[inline(never)]
fn init_cold() -> Result<()> {
    unsafe { Ok(sys::optixInit().to_result()?) }
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

/// Launch the given [`Pipeline`](pipeline::Pipeline) on the given [`Stream`](cust::stream::Stream).
///
/// # Safety
/// You must ensure that:
/// - Any [`ProgramGroup`](program_group::ProgramGroup)s referenced by the [`Pipeline`](pipeline::Pipeline) are still alive
/// - Any device memory referenced in `buf_launch_params` point to valid,
///   correctly aligned memory
/// - Any [`SbtRecord`](shader_binding_table::SbtRecord)s and associated data referenced by the
///   [`ShaderBindingTable`](shader_binding_table::ShaderBindingTable) are alive and valid
pub unsafe fn launch<P: cust::memory::DeviceCopy>(
    pipeline: &crate::pipeline::Pipeline,
    stream: &cust::stream::Stream,
    buf_launch_params: &mut cust::memory::DeviceBox<P>,
    sbt: &sys::OptixShaderBindingTable,
    width: u32,
    height: u32,
    depth: u32,
) -> Result<()> {
    Ok(optix_call!(optixLaunch(
        pipeline.raw,
        stream.as_inner(),
        buf_launch_params.as_device_ptr().as_raw() as u64,
        std::mem::size_of::<P>(),
        sbt,
        width,
        height,
        depth,
    ))?)
}

#[cfg(feature = "glam")]
mod impl_glam;

macro_rules! const_assert {
    ($x:expr $(,)?) => {
        #[allow(unknown_lints, clippy::eq_op)]
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}
pub(crate) use const_assert;

macro_rules! const_assert_eq {
    ($x:expr, $y:expr $(,)?) => {
        const_assert!($x == $y);
    };
}
pub(crate) use const_assert_eq;
