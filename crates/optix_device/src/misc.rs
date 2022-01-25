use cuda_std::gpu_only;

#[gpu_only]
pub fn primitive_index() -> u32 {
    let mut idx: u32;
    unsafe {
        asm!("call ({}), _optix_read_primitive_idx, ();", out(reg32) idx);
    }
    idx
}
