use libc::c_char;

/// Extension trait for explicit casts to `*const c_char`.
pub(crate) trait AsCCharPtr {
    /// Equivalent to `self.as_ptr().cast()`, but only casts to `*const c_char`.
    fn as_c_char_ptr(&self) -> *const c_char;
}

impl AsCCharPtr for str {
    fn as_c_char_ptr(&self) -> *const c_char {
        self.as_ptr().cast()
    }
}

impl AsCCharPtr for [u8] {
    fn as_c_char_ptr(&self) -> *const c_char {
        self.as_ptr().cast()
    }
}
