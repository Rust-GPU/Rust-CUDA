pub fn pack_pointer<T>(ptr: *mut T) -> (u32, u32) {
    let x = ptr as u32;
    let y = (ptr as u64 >> 32) as u32;
    (x, y)
}

pub fn unpack_pointer<T>(x: u32, y: u32) -> *mut T {
    (((y as u64) << 32) | x as u64) as *mut T
}
