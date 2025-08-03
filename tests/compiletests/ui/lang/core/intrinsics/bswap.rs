// Test bswap intrinsic
// build-pass

#![allow(internal_features)]
#![feature(core_intrinsics)]
#![no_std]

use core::intrinsics::bswap;
use cuda_std::kernel;

#[kernel]
pub unsafe fn test_bswap() {
    let original_i16: i16 = 0x1234;
    let swapped_i16 = bswap(original_i16);

    let original_neg_i16: i16 = -0x1234;
    let swapped_neg_i16 = bswap(original_neg_i16);

    let original_i32: i32 = 0x12345678;
    let swapped_i32 = bswap(original_i32);

    let original_neg_i32: i32 = -0x12345678;
    let swapped_neg_i32 = bswap(original_neg_i32);

    let original_zero_i16: i16 = 0;
    let swapped_zero_i16 = bswap(original_zero_i16);

    let original_zero_i32: i32 = 0;
    let swapped_zero_i32 = bswap(original_zero_i32);

    let original_u8: u8 = 0x12;
    let swapped_u8 = bswap(original_u8);

    let original_u16: u16 = 0x1234;
    let swapped_u16 = bswap(original_u16);

    let original_u32: u32 = 0x12345678;
    let swapped_u32 = bswap(original_u32);

    let original_u64: u64 = 0x123456789ABCDEF0;
    let swapped_u64 = bswap(original_u64);
}
