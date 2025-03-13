use crate::{sys::v2::*, BlasDatatype};
use num_complex::{Complex32, Complex64};
use std::os::raw::c_int;

pub trait Level1: BlasDatatype {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t;
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t;
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t;
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self::FloatTy,
    ) -> cublasStatus_t;
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t;
    unsafe fn rotm(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        param: *const Self::FloatTy,
    ) -> cublasStatus_t;
    unsafe fn rotmg(
        handle: cublasHandle_t,
        d1: *mut Self,
        d2: *mut Self,
        x1: *mut Self,
        y1: *const Self,
        param: *mut Self,
    ) -> cublasStatus_t;
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t;
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t;
}

impl Level1 for f32 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIsamax_v2(handle, n, x, incx, result)
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIsamin_v2(handle, n, x, incx, result)
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasScopy_v2(handle, n, x, incx, y, incy)
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        cublasSnrm2_v2(handle, n, x, incx, result)
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self,
    ) -> cublasStatus_t {
        cublasSrot_v2(handle, n, x, incx, y, incy, c, s)
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        cublasSrotg_v2(handle, a, b, c, s)
    }
    unsafe fn rotm(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        cublasSrotm_v2(handle, n, x, incx, y, incy, param)
    }
    unsafe fn rotmg(
        handle: cublasHandle_t,
        d1: *mut Self,
        d2: *mut Self,
        x1: *mut Self,
        y1: *const Self,
        param: *mut Self,
    ) -> cublasStatus_t {
        cublasSrotmg_v2(handle, d1, d2, x1, y1, param)
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        cublasSscal_v2(handle, n, alpha, x, incx)
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasSswap_v2(handle, n, x, incx, y, incy)
    }
}

impl Level1 for f64 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIdamax_v2(handle, n, x, incx, result)
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIdamin_v2(handle, n, x, incx, result)
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy)
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasDcopy_v2(handle, n, x, incx, y, incy)
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        cublasDnrm2_v2(handle, n, x, incx, result)
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self,
    ) -> cublasStatus_t {
        cublasDrot_v2(handle, n, x, incx, y, incy, c, s)
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        cublasDrotg_v2(handle, a, b, c, s)
    }
    unsafe fn rotm(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        cublasDrotm_v2(handle, n, x, incx, y, incy, param)
    }
    unsafe fn rotmg(
        handle: cublasHandle_t,
        d1: *mut Self,
        d2: *mut Self,
        x1: *mut Self,
        y1: *const Self,
        param: *mut Self,
    ) -> cublasStatus_t {
        cublasDrotmg_v2(handle, d1, d2, x1, y1, param)
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        cublasDscal_v2(handle, n, alpha, x, incx)
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasDswap_v2(handle, n, x, incx, y, incy)
    }
}

impl Level1 for Complex32 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIcamax_v2(handle, n, x.cast(), incx, result)
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIcamin_v2(handle, n, x.cast(), incx, result)
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasCaxpy_v2(handle, n, alpha.cast(), x.cast(), incx, y.cast(), incy)
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasCcopy_v2(handle, n, x.cast(), incx, y.cast(), incy)
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        cublasScnrm2_v2(handle, n, x.cast(), incx, result)
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self::FloatTy,
    ) -> cublasStatus_t {
        cublasCsrot_v2(handle, n, x.cast(), incx, y.cast(), incy, c, s)
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        cublasCrotg_v2(handle, a.cast(), b.cast(), c, s.cast())
    }
    unsafe fn rotm(
        _handle: cublasHandle_t,
        _n: c_int,
        _x: *mut Self,
        _incx: c_int,
        _y: *mut Self,
        _incy: c_int,
        _param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn rotmg(
        _handle: cublasHandle_t,
        _d1: *mut Self,
        _d2: *mut Self,
        _x1: *mut Self,
        _y1: *const Self,
        _param: *mut Self,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        cublasCscal_v2(handle, n, alpha.cast(), x.cast(), incx)
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasCswap_v2(handle, n, x.cast(), incx, y.cast(), incy)
    }
}

impl Level1 for Complex64 {
    unsafe fn amax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIzamax_v2(handle, n, x.cast(), incx, result)
    }
    unsafe fn amin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t {
        cublasIzamin_v2(handle, n, x.cast(), incx, result)
    }
    unsafe fn axpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasZaxpy_v2(handle, n, alpha.cast(), x.cast(), incx, y.cast(), incy)
    }
    unsafe fn copy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasZcopy_v2(handle, n, x.cast(), incx, y.cast(), incy)
    }
    unsafe fn nrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        result: *mut Self::FloatTy,
    ) -> cublasStatus_t {
        cublasDznrm2_v2(handle, n, x.cast(), incx, result)
    }
    unsafe fn rot(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
        c: *const Self::FloatTy,
        s: *const Self::FloatTy,
    ) -> cublasStatus_t {
        cublasZdrot_v2(handle, n, x.cast(), incx, y.cast(), incy, c, s)
    }
    unsafe fn rotg(
        handle: cublasHandle_t,
        a: *mut Self,
        b: *mut Self,
        c: *mut Self::FloatTy,
        s: *mut Self,
    ) -> cublasStatus_t {
        cublasZrotg_v2(handle, a.cast(), b.cast(), c, s.cast())
    }
    unsafe fn rotm(
        _handle: cublasHandle_t,
        _n: c_int,
        _x: *mut Self,
        _incx: c_int,
        _y: *mut Self,
        _incy: c_int,
        _param: *const Self::FloatTy,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn rotmg(
        _handle: cublasHandle_t,
        _d1: *mut Self,
        _d2: *mut Self,
        _x1: *mut Self,
        _y1: *const Self,
        _param: *mut Self,
    ) -> cublasStatus_t {
        unreachable!()
    }
    unsafe fn scal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const Self,
        x: *mut Self,
        incx: c_int,
    ) -> cublasStatus_t {
        cublasZscal_v2(handle, n, alpha.cast(), x.cast(), incx)
    }
    unsafe fn swap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut Self,
        incx: c_int,
        y: *mut Self,
        incy: c_int,
    ) -> cublasStatus_t {
        cublasZswap_v2(handle, n, x.cast(), incx, y.cast(), incy)
    }
}

/// Level-1 Methods exclusive to complex numbers.
pub trait ComplexLevel1: BlasDatatype {
    unsafe fn dotu(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t;
    unsafe fn dotc(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t;
}

impl ComplexLevel1 for Complex32 {
    unsafe fn dotu(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        cublasCdotu_v2(handle, n, x.cast(), incx, y.cast(), incy, result.cast())
    }
    unsafe fn dotc(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        cublasCdotc_v2(handle, n, x.cast(), incx, y.cast(), incy, result.cast())
    }
}

impl ComplexLevel1 for Complex64 {
    unsafe fn dotu(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        cublasZdotu_v2(handle, n, x.cast(), incx, y.cast(), incy, result.cast())
    }
    unsafe fn dotc(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        cublasZdotc_v2(handle, n, x.cast(), incx, y.cast(), incy, result.cast())
    }
}

/// Level-1 Methods exclusive to floats.
pub trait FloatLevel1: BlasDatatype {
    unsafe fn dot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t;
}

impl FloatLevel1 for f32 {
    unsafe fn dot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        cublasSdot_v2(handle, n, x, incx, y, incy, result)
    }
}

impl FloatLevel1 for f64 {
    unsafe fn dot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const Self,
        incx: c_int,
        y: *const Self,
        incy: c_int,
        result: *mut Self,
    ) -> cublasStatus_t {
        cublasDdot_v2(handle, n, x, incx, y, incy, result)
    }
}
