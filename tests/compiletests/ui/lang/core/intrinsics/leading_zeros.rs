// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_leading_zeros() {
    // 8-bit tests
    assert!(0b00000000_u8.leading_zeros() == 8);
    assert!(0b00000001_u8.leading_zeros() == 7);
    assert!(0b00010000_u8.leading_zeros() == 3);
    assert!(0b10000000_u8.leading_zeros() == 0);
    assert!(0b11111111_u8.leading_zeros() == 0);

    // 16-bit tests
    assert!(0x0000_u16.leading_zeros() == 16);
    assert!(0x0001_u16.leading_zeros() == 15);
    assert!(0x0100_u16.leading_zeros() == 7);
    assert!(0x8000_u16.leading_zeros() == 0);
    assert!(0xFFFF_u16.leading_zeros() == 0);

    // 32-bit tests
    assert!(0x00000000_u32.leading_zeros() == 32);
    assert!(0x00000001_u32.leading_zeros() == 31);
    assert!(0x00010000_u32.leading_zeros() == 15);
    assert!(0x80000000_u32.leading_zeros() == 0);
    assert!(0xFFFFFFFF_u32.leading_zeros() == 0);

    // 64-bit tests
    assert!(0x0000000000000000_u64.leading_zeros() == 64);
    assert!(0x0000000000000001_u64.leading_zeros() == 63);
    assert!(0x8000000000000000_u64.leading_zeros() == 0);
    assert!(0xFFFFFFFFFFFFFFFF_u64.leading_zeros() == 0);
    assert!(0x00000000_12345678_u64.leading_zeros() == 32);
    assert!(0x00000000_80000000_u64.leading_zeros() == 32);
    assert!(0x00100000_00000000_u64.leading_zeros() == 11);
    assert!(0x12345678_00000000_u64.leading_zeros() == 3);

    // Signed integers (should behave the same as unsigned)
    assert!(0i8.leading_zeros() == 8);
    assert!((-1i8).leading_zeros() == 0);
    assert!(0i16.leading_zeros() == 16);
    assert!((-1i16).leading_zeros() == 0);
    assert!(0i32.leading_zeros() == 32);
    assert!((-1i32).leading_zeros() == 0);
    assert!(0i64.leading_zeros() == 64);
    assert!((-1i64).leading_zeros() == 0);
}
