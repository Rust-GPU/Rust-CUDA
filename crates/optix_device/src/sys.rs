#![allow(clippy::missing_safety_doc)]

use crate::trace::{RayFlags, TraversableHandle};
#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::gpu_only;
use glam::Vec3;
use paste::paste;

macro_rules! set_payload {
    ($($number:literal),*) => {
        paste! {
            $(
                #[gpu_only]
                pub unsafe fn [<set_payload_ $number>](p: u32) {
                    asm!(
                        "call _optix_set_payload, ({}, {});",
                        in(reg32) $number,
                        in(reg32) p
                    )
                }
            )*

            #[gpu_only]
            pub unsafe fn set_payload(idx: u8, p: u32) {
                match idx {
                    $(
                        $number => [<set_payload_ $number>](p)
                    ),*,
                    _ => panic!("Too many registers used!")
                }
            }
        }
   };
}

set_payload! {
    0, 1, 2, 3, 4, 5, 6, 7
}

macro_rules! get_payload {
    ($($number:literal),*) => {
        paste! {
            $(
                #[gpu_only]
                pub unsafe fn [<get_payload_ $number>]() -> u32 {
                    let p: u32;
                    asm!(
                        "call ({}), _optix_get_payload, ({});",
                        out(reg32) p,
                        in(reg32) $number,
                    );
                    p
                }
            )*

            #[gpu_only]
            pub unsafe fn get_payload(idx: u8) -> u32 {
                match idx {
                    $(
                        $number => [<get_payload_ $number>]()
                    ),*,
                    _ => panic!("Too many registers used!")
                }
            }
        }
   };
}

get_payload! {
    0, 1, 2, 3, 4, 5, 6, 7
}

macro_rules! trace_fns {
    ($($num:tt ($($used:ident)*) $($unused:ident)*),* $(,)?) => {
        $(
            paste! {
                #[gpu_only]
                #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
                pub unsafe fn [<trace_ $num>](
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
                    $(
                        $used: &mut u32
                    ),*
                ) {
                    let [ox, oy, oz] = ray_origin.to_array();
                    let [dx, dy, dz] = ray_direction.to_array();
                    let handle = handle.0;
                    let ray_flags = ray_flags.bits();
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
                        $(
                            $used = inout(reg32) *$used,
                        )*
                        $(
                            $unused = out(reg32) _
                        ),*
                    );
                }
            }
        )*
    };
}

// i know this is ugly but it would have taken more time to write a sophisticated macro for this. Besides, this looks a lot
// like a graph for my sanity when writing this 📉
trace_fns! {
    0   ()p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    1    (p0)p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    2    (p0 p1)p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    3    (p0 p1 p2)p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    4    (p0 p1 p2 p3)p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    5    (p0 p1 p2 p3 p4)p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    6    (p0 p1 p2 p3 p4 p5)p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    7    (p0 p1 p2 p3 p4 p5 p6)p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    8    (p0 p1 p2 p3 p4 p5 p6 p7)p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
    9    (p0 p1 p2 p3 p4 p5 p6 p7 p8)p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   10    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9)p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   11    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10)p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   12    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11)p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   13    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12)p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   14    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13)p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   15    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14)p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   16    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15)p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   17    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16)p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   18    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17)p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   19    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18)p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   20    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19)p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   21    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20)p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   22    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21)p22 p23 p24 p25 p26 p27 p28 p29 p30 p31,
   23    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22)p23 p24 p25 p26 p27 p28 p29 p30 p31,
   24    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23)p24 p25 p26 p27 p28 p29 p30 p31,
   25    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24)p25 p26 p27 p28 p29 p30 p31,
   26    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25)p26 p27 p28 p29 p30 p31,
   27    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26)p27 p28 p29 p30 p31,
   28    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27)p28 p29 p30 p31,
   29    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28)p29 p30 p31,
   30    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29)p30 p31,
   31    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30)p31,
   32    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31),
}
