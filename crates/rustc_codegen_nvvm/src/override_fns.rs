//! Functions for overriding certain functions in certain crates with special
//! codegen-builtin methods. Currently the only use for this is overriding libm functions
//! with libdevice intrinsics (which are much faster and smaller).

use crate::{builder::Builder, context::CodegenCx, llvm};
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::BuilderMethods;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::{mir::mono::MonoItem, ty::Instance};

/// Either override or define a function.
pub(crate) fn define_or_override_fn<'ll, 'tcx>(func: Instance<'tcx>, cx: &CodegenCx<'ll, 'tcx>) {
    if should_override(func, cx) {
        override_libm_function(func, cx);
    } else {
        MonoItem::define::<Builder<'_, '_, '_>>(&MonoItem::Fn(func), cx);
    }
}

fn should_override<'ll, 'tcx>(func: Instance<'tcx>, cx: &CodegenCx<'ll, 'tcx>) -> bool {
    if !cx.codegen_args.override_libm {
        return false;
    }

    // there is no better way to do this without putting some sort of diagnostic/lang item in libm
    let is_libm = cx.tcx.crate_name(LOCAL_CRATE).as_str() == "libm";
    if !is_libm {
        return false;
    }
    let sym = cx.tcx.item_name(func.def_id());
    let name = sym.as_str();
    let intrinsics = cx.intrinsics_map.borrow();
    let is_known_intrinsic = intrinsics.contains_key(format!("__nv_{}", name).as_str());

    !is_unsupported_libdevice_fn(&name) && is_known_intrinsic
}

fn is_unsupported_libdevice_fn(name: &str) -> bool {
    // libm functions for which libdevice has no intrinsics for.
    const UNSUPPORTED: &[&str] = &[
        // doesnt exist
        "lgamma_r",
        "lgammaf_r",
        // different signatures
        "sincos",
        "sincosf",
    ];
    UNSUPPORTED.contains(&name)
}

fn override_libm_function<'ll, 'tcx>(func: Instance<'tcx>, cx: &CodegenCx<'ll, 'tcx>) {
    let name = cx.tcx.item_name(func.def_id());
    let nv_name = format!("__nv_{}", name.as_str());
    let (intrinsic_llfn_ty, intrinsic_llfn) = cx.get_intrinsic(nv_name.as_str());

    let llfn = cx.get_fn(func);
    let start = Builder::append_block(cx, llfn, "start");
    let mut bx = Builder::build(cx, start);

    let params = llvm::get_params(llfn);
    let llcall = bx.call(intrinsic_llfn_ty, None, None, intrinsic_llfn, &params, None, None);
    bx.ret(llcall);
}
