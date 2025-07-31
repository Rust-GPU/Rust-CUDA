#[cfg_attr(target_os = "cuda", panic_handler)]
#[allow(dead_code)]
fn panic(_: &core::panic::PanicInfo) -> ! {
    #[allow(clippy::empty_loop)]
    loop {}
}
