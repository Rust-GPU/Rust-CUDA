use cuda_std::kernel;

#[kernel]
pub unsafe fn bad_kernel() {
    let _s = std::fs::File::create("foo.txt");
}
