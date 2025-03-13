#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::gpu_only;
use paste::paste;
use seq_macro::seq;

#[gpu_only]
pub fn primitive_index() -> u32 {
    let mut idx: u32;
    unsafe {
        asm!("call ({}), _optix_read_primitive_idx, ();", out(reg32) idx);
    }
    idx
}

pub trait IntersectionPayload {
    fn report_intersection(hit_t: f32, hit_kind: u8, payload: Self) -> bool;
}

macro_rules! impl_intersection_payload {
    ($num:tt) => {
        paste! {
            impl IntersectionPayload for [u32; $num] {
                #[gpu_only]
                fn report_intersection(
                    hit_t: f32,
                    hit_kind: u8,
                    payload: Self,
                ) -> bool {
                    seq!(
                        P in 0..$num {{
                            let [#(p~P,)*] = payload;
                            let out: u32;
                            unsafe {
                                asm!(
                                    "{{",
                                    ".reg .f32 %f0;",
                                    "mov.f32 %f0, {hit_t};",
                                    concat!("call ({}), _optix_report_intersection_", stringify!($num)),
                                    concat!(", (%f0, {}", #(concat!(", {", stringify!(p~P), "}"),)* ");"),
                                    "}}",
                                    out(reg32) out,
                                    in(reg32) hit_kind,
                                    hit_t = in(reg32) hit_t,
                                    #(
                                        p~P = in(reg32) p~P,
                                    )*
                                );
                            }
                            out != 0
                        }}
                    )
                }
            }
        }
    };
    () => {
        seq! {
            N in 0..=7 {
                impl_intersection_payload! { N }
            }
        }
    }
}

impl_intersection_payload! {}

/// Reports an intersection and passes custom attributes to further programs.
///
/// If `tmin <= hit_t <= tmax` then the anyhit program associated with this intersection
/// will be invoked, then the program will do one of three things:
///  - Ignore the intersection; no hit is recorded and this function returns `false`.
///  - Terminate the ray; a hit is recorded and this function does not return. No
///    further traversal occurs and the associated closesthit program is invoked.
///  - Neither; A hit is recorded and this function returns `true`.
///
/// **Only the lower 7 bits of the `hit_kind` should be written, the top 127 values are
/// reserved for hardware primitives.**
pub fn report_intersection<P: IntersectionPayload>(hit_t: f32, hit_kind: u8, payload: P) -> bool {
    P::report_intersection(hit_t, hit_kind, payload)
}

/// Records the hit, stops traversal, then proceeds to the closesthit program.
#[gpu_only]
pub fn terminate_ray() {
    unsafe {
        asm!("call _optix_terminate_ray, ();");
    }
}

/// Discards the hit and returns control to the calling intersection program or the built-in intersection hardware.
#[gpu_only]
pub fn ignore_intersection() {
    unsafe {
        asm!("call _optix_ignore_intersection, ();");
    }
}

macro_rules! get_attribute_fns {
    ($num:tt) => {
        paste! {
            #[gpu_only]
            #[allow(clippy::missing_safety_doc)]
            unsafe fn [<get_attribute_ $num>]() -> u32 {
                let out: u32;
                asm!(
                    concat!("call ({}), _optix_get_attribute_", stringify!($num), ", ();"),
                    out(reg32) out
                );
                out
            }
        }
    };
    () => {
        seq! {
            N in 0..=7 {
                get_attribute_fns! { N }
            }
        }
    };
}

get_attribute_fns! {}

/// Retrieves an attribute set by the intersection program when reporting an intersection.
///
/// # Safety
///
/// The attribute must have been set by the intersection program.
///
/// # Panics
///
/// Panics if the idx is over `7`.
pub unsafe fn get_attribute(idx: u8) -> u32 {
    match idx {
        0 => get_attribute_0(),
        1 => get_attribute_1(),
        2 => get_attribute_2(),
        3 => get_attribute_3(),
        4 => get_attribute_4(),
        5 => get_attribute_5(),
        6 => get_attribute_6(),
        7 => get_attribute_7(),
        _ => panic!("Invalid attribute index"),
    }
}
