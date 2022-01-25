use cuda_std::gpu_only;
use glam::Vec3;

/// An opaque handle to a traversable BVH.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraversableHandle(u64);

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

#[gpu_only]
#[allow(clippy::too_many_arguments)]
pub unsafe fn trace(
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
    p0: &mut u32,
    p1: &mut u32,
) {
    let [ox, oy, oz] = ray_origin.to_array();
    let [dx, dy, dz] = ray_direction.to_array();
    let handle = handle.0;
    let ray_flags = ray_flags.bits;
    let visibility_mask = visbility_mask as u32;

    // NOTE(RDambrosio016): This is horrific, so let me take a second to explain. OptiX is entirely just
    // inline assembly, which makes sense, that is what a gpu-side library is. Which is great, we can just look
    // at the internal headers and see how optix works. This works fine, but for some strange reason, optix really
    // wants floats to be passed as .f32 regs in _optix_trace_typed_32. Except rust inline asm doesn't support float regs
    // for nvptx. So we declare our own regs then mov our b32 regs into it. If we do not do this, optix will stack overflow
    // when trying to make a module, why? i have no idea.
    asm!(
        "{{",
            ".reg .f32 %f<9>;",
            "mov.f32 	%f0, {ox};",
            "mov.f32 	%f1, {oy};",
            "mov.f32 	%f2, {oz};",
            "mov.f32 	%f3, {dx};",
            "mov.f32 	%f4, {dy};",
            "mov.f32 	%f5, {dz};",
            "mov.f32 	%f6, {tmin};",
            "mov.f32 	%f7, {tmax};",
            "mov.f32 	%f8, {ray_time};",
            concat!(
            "call",
            "({p0}, {p1}, {p2}, {p3}, {p4}, {p5}, {p6}, {p7}, {p8}, {p9}, {p10},",
            "{p11}, {p12}, {p13}, {p14}, {p15}, {p16}, {p17}, {p18}, {p19}, {p20}, {p21},",
            "{p22}, {p23}, {p24}, {p25}, {p26}, {p27}, {p28}, {p29}, {p30}, {p31}),",
            "_optix_trace_typed_32,",
            "({}, {}, %f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8, {}, {}, {}, {}, {}, {},",
            "{p0}, {p1}, {p2}, {p3}, {p4}, {p5}, {p6}, {p7}, {p8}, {p9}, {p10},",
            "{p11}, {p12}, {p13}, {p14}, {p15}, {p16}, {p17}, {p18}, {p19}, {p20}, {p21},",
            "{p22}, {p23}, {p24}, {p25}, {p26}, {p27}, {p28}, {p29}, {p30}, {p31});",
            ),
        "}}",
        in(reg32) 0,
        in(reg64) handle,
        in(reg32) visibility_mask,
        in(reg32) ray_flags,
        in(reg32) sbt_offset,
        in(reg32) sbt_stride,
        in(reg32) miss_sbt_index,
        in(reg32) 2,
        ox = in(reg32) ox,
        oy = in(reg32) oy,
        oz = in(reg32) oz,
        dx = in(reg32) dx,
        dy = in(reg32) dy,
        dz = in(reg32) dz,
        tmin = in(reg32) tmin,
        tmax = in(reg32) tmax,
        ray_time = in(reg32) ray_time,
        p0 = inout(reg32) *p0,
        p1 = inout(reg32) *p1,
        p2 = out(reg32) _,
        p3 = out(reg32) _,
        p4 = out(reg32) _,
        p5 = out(reg32) _,
        p6 = out(reg32) _,
        p7 = out(reg32) _,
        p8 = out(reg32) _,
        p9 = out(reg32) _,
        p10 = out(reg32) _,
        p11 = out(reg32) _,
        p12 = out(reg32) _,
        p13 = out(reg32) _,
        p14 = out(reg32) _,
        p15 = out(reg32) _,
        p16 = out(reg32) _,
        p17 = out(reg32) _,
        p18 = out(reg32) _,
        p19 = out(reg32) _,
        p20 = out(reg32) _,
        p21 = out(reg32) _,
        p22 = out(reg32) _,
        p23 = out(reg32) _,
        p24 = out(reg32) _,
        p25 = out(reg32) _,
        p26 = out(reg32) _,
        p27 = out(reg32) _,
        p28 = out(reg32) _,
        p29 = out(reg32) _,
        p30 = out(reg32) _,
        p31 = out(reg32) _,
    );
}
