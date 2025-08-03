// build-pass

use cuda_std::kernel;

struct A;
struct B;

struct S {
    x: A,
    y: B,
}

fn f(x: &B) {}

#[kernel]
pub unsafe fn test_zst_member_ref_arg() {
    let s = S { x: A, y: B };
    f(&s.y);
}
