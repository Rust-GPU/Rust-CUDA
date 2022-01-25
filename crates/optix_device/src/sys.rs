#![allow(clippy::missing_safety_doc)]

use cuda_std::gpu_only;
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
