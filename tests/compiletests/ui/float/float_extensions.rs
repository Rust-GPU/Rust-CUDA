// Test CUDA float extension functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::FloatExt;

#[kernel]
pub unsafe fn test_float_extensions() {
    let x = 3.14f32;

    // Test various float extension methods
    let _cospi = x.cospi();
    let _erf = x.error_function();
    let _erfc = x.complementary_error_function();
    let _erfcx = x.scaled_complementary_error_function();

    // Test frexp
    let (_frac, _exp) = x.frexp();
    let _exp = x.unbiased_exp();

    // Test bessel functions
    let _j0 = x.j0();
    let _j1 = x.j1();
    let _jn = x.jn(2);

    // Test other functions
    let _ldexp = x.ldexp(3);
    let _lgamma = x.log_gamma();
    let _log1p = x.log1p();

    // Test normcdf functions
    let _normcdf = x.norm_cdf();
    let _normcdfinv = (0.5f32).inv_norm_cdf();

    // Test sinpi
    let _sinpi = x.sinpi();

    // Test f64 as well
    let y = 2.718f64;
    let _cospi_f64 = y.cospi();
    let _erf_f64 = y.error_function();
}
