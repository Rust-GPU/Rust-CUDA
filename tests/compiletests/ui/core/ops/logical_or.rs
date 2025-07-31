// Test using `||` operator.
// build-pass

use cuda_std::kernel;

fn f(x: bool, y: bool) -> bool {
    x || y
}

#[kernel]
pub unsafe fn main() {
    f(false, true);
}
