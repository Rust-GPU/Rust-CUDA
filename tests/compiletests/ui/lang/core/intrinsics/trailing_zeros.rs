// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_trailing_zeros() {
    // 8-bit tests
    assert!(0b00000000_u8.trailing_zeros() == 8);
    assert!(0b00000001_u8.trailing_zeros() == 0);
    assert!(0b00010000_u8.trailing_zeros() == 4);
    assert!(0b10000000_u8.trailing_zeros() == 7);
    assert!(0b11111110_u8.trailing_zeros() == 1);

    // 16-bit tests
    assert!(0x0000_u16.trailing_zeros() == 16);
    assert!(0x0001_u16.trailing_zeros() == 0);
    assert!(0x0100_u16.trailing_zeros() == 8);
    assert!(0x8000_u16.trailing_zeros() == 15);
    assert!(0xFFFE_u16.trailing_zeros() == 1);

    // 32-bit tests
    assert!(0x00000000_u32.trailing_zeros() == 32);
    assert!(0x00000001_u32.trailing_zeros() == 0);
    assert!(0x00010000_u32.trailing_zeros() == 16);
    assert!(0x80000000_u32.trailing_zeros() == 31);
    assert!(0xFFFFFFFE_u32.trailing_zeros() == 1);

    // 64-bit tests
    assert!(0x0000000000000000_u64.trailing_zeros() == 64);
    assert!(0x0000000000000001_u64.trailing_zeros() == 0);
    assert!(0x8000000000000000_u64.trailing_zeros() == 63);
    assert!(0xFFFFFFFFFFFFFFFE_u64.trailing_zeros() == 1);
    assert!(0x12340000_00000000_u64.trailing_zeros() == 32);
    assert!(0x00000001_00000000_u64.trailing_zeros() == 32);
    assert!(0x00000000_00001000_u64.trailing_zeros() == 12);
    assert!(0x00000000_80000000_u64.trailing_zeros() == 31);

    // Signed integers (should behave the same as unsigned)
    assert!(0i8.trailing_zeros() == 8);
    assert!((-1i8).trailing_zeros() == 0);
    assert!(0i16.trailing_zeros() == 16);
    assert!((-1i16).trailing_zeros() == 0);
    assert!(0i32.trailing_zeros() == 32);
    assert!((-1i32).trailing_zeros() == 0);
    assert!(0i64.trailing_zeros() == 64);
    assert!((-1i64).trailing_zeros() == 0);
}
