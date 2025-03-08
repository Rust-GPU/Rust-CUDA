use crate::sys::*;
use cust_core::DeviceCopy;
use glam::Vec3;
use paste::paste;
use seq_macro::seq;

/// An opaque handle to a traversable BVH.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, DeviceCopy)]
pub struct TraversableHandle(pub(crate) u64);

impl TraversableHandle {
    /// A null traversable handle which will cause only the miss program to be invoked.
    pub fn null() -> TraversableHandle {
        TraversableHandle(0)
    }
}

bitflags::bitflags! {
    /// Ray flags passed when tracing rays to control their behavior.
    pub struct RayFlags: u32 {
        /// Disables anyhit programs for the ray. Mutually exclusive with
        /// [`RayFlags::ENABLE_ANYHIT`], [`RayFlags::CULL_DISABLED_ANYHIT`], and [`RayFlags::CULL_ENFORCED_ANYHIT`].
        const DISABLE_ANYHIT = 1;
        /// Forces anyhit execution for the ray. Mutually exclusive with
        /// [`RayFlags::DISABLE_ANYHIT`], [`RayFlags::CULL_DISABLED_ANYHIT`], and [`RayFlags::CULL_ENFORCED_ANYHIT`].
        const ENFORCE_ANYHIT = 1 << 1;
        /// Terminates the ray after the first hit and executes the closesthit
        /// program of that hit.
        const TERMINATE_ON_FIRST_HIT = 1 << 2;
        /// Disables the closesthit programs for the ray, but still executes the miss
        /// program in case of a miss.
        const DISABLE_CLOSESTHIT = 1 << 3;
        /// Do not intersect back faces, respecting instance triangle-flip options.
        /// Mutually exclusive with [`RayFlags::CULL_FRONT_FACING_TRIANGLES`].
        const CULL_BACK_FACING_TRIANGLES = 1 << 4;
        /// Do not intersect front faces, respecting instance triangle-flip options.
        /// Mutually exclusive with [`RayFlags::CULL_BACK_FACING_TRIANGLES`].
        const CULL_FRONT_FACING_TRIANGLES = 1 << 5;
        /// Do not intersect geometry which disables anyhit programs on it.
        /// Mutually exclusive with [`RayFlags::CULL_ENFORCED_ANYHIT`], [`RayFlags::ENFORCE_ANYHIT`], and
        /// [`RayFlags::DISABLE_ANYHIT`].
        const CULL_DISABLED_ANYHIT = 1 << 6;
        /// Do not intersect geometry which enables anyhit programs on it.
        /// Mutually exclusive with [`RayFlags::CULL_DISABLED_ANYHIT`], [`RayFlags::ENFORCE_ANYHIT`], and
        /// [`RayFlags::DISABLE_ANYHIT`].
        const CULL_ENFORCED_ANYHIT = 1 << 7;
    }
}

/// Instantiates a new ray trace into the scene.
///
/// The hardware will perform a query into the scene graph in hardware, running any
/// required programs as necessary. For example, running the `miss` program if the ray
/// missed any intersectable object in the scene. Running the `anyhit` and `closesthit`
/// programs if specified, etc.
///
/// # Parameters
///
///  - `handle`: a valid handle to a traversable BVH, usually stored in the launch
///    params and populated by the host.
///   - `ray_origin`: the origin of the ray.
///   - `ray_direction`: the direction of the ray.
///   - `tmin`: the minimum distance along the ray to accept as a hit.
///   - `tmax`: the maximum distance along the ray to accept as a hit.
///   - `ray_time`: the time allocated for motion-aware traversal and material
///     evaluation. If motion is not enabled in the pipeline compiler options, this
///     argument is ignored.
///   - `ray_flags`: a set of flags to control the behavior of the ray.
///   - `visibility_mask`: Controls intersection across configurable masks of instances.
///     An intersection for an instance will only be computed if there is at least a
///     matching bit in both masks. Usually set to `255` if this masking isn't being
///     used.
///   - `sbt_offset`: A valid offset into the shader binding table for selecting the sbt
///     record for a ray intersection.
///   - `sbt_stride`: A valid stride into the shader binding table for selecting the sbt
///     record for a ray intersection.
///   - `miss_sbt_index`: The index of the sbt record to use for the miss program.
///   - `payload`: An array of references to 32-bit values for the payload of the ray.
///
/// # Notes
///
/// - **OptiX rejects values of infinity for `tmin` and `tmax` and it will throw an
///   exception, use [`f32::MAX`] instead**
///
/// # Safety
///
/// `sbt_offset`, `sbt_stride`, and `miss_sbt_index` must be valid
/// offsets/strides/indices for the shader binding table.
#[allow(clippy::too_many_arguments)]
pub unsafe fn trace<P: TracePayload>(
    handle: TraversableHandle,
    ray_origin: Vec3,
    ray_direction: Vec3,
    tmin: f32,
    tmax: f32,
    ray_time: f32,
    visbility_mask: u8,
    ray_flags: RayFlags,
    sbt_offset: u32,
    sbt_stride: u32,
    miss_sbt_index: u32,
    payload: P,
) {
    P::trace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        ray_time,
        visbility_mask,
        ray_flags,
        sbt_offset,
        sbt_stride,
        miss_sbt_index,
        payload,
    )
}

pub trait TracePayload {
    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    unsafe fn trace(
        handle: TraversableHandle,
        ray_origin: Vec3,
        ray_direction: Vec3,
        tmin: f32,
        tmax: f32,
        ray_time: f32,
        visbility_mask: u8,
        ray_flags: RayFlags,
        sbt_offset: u32,
        sbt_stride: u32,
        miss_sbt_index: u32,
        payload: Self,
    );
}

macro_rules! impl_trace_payload_array {
    ($($num:tt),*) => {
        paste! {
            $(
                impl TracePayload for [&mut u32; $num] {
                    unsafe fn trace(
                        handle: TraversableHandle,
                        ray_origin: Vec3,
                        ray_direction: Vec3,
                        tmin: f32,
                        tmax: f32,
                        ray_time: f32,
                        visbility_mask: u8,
                        ray_flags: RayFlags,
                        sbt_offset: u32,
                        sbt_stride: u32,
                        miss_sbt_index: u32,
                        payload: Self,
                    ) {
                        seq!(
                            P in 0..$num {{
                                let [#(p~P,)*] = payload;
                                [<trace_ $num>](
                                handle,
                                ray_origin,
                                ray_direction,
                                tmin,
                                tmax,
                                ray_time,
                                visbility_mask,
                                ray_flags,
                                sbt_offset,
                                sbt_stride,
                                miss_sbt_index,
                                #(
                                    p~P,
                                )*
                                )
                            }}
                        )
                    }
                }
            )*
        }
    };
}

impl_trace_payload_array! {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
}
