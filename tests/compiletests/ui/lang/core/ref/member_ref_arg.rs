// build-pass

use cuda_std::kernel;

struct S {
    x: u32,
    y: u32,
}

#[inline(never)]
fn f(x: &u32) {}

#[inline(never)]
fn g(xy: (&u32, &u32)) {}

#[kernel]
pub unsafe fn test_member_ref_arg() {
    let s = S { x: 2, y: 2 };
    f(&s.x);
    g((&s.x, &s.y));
}
