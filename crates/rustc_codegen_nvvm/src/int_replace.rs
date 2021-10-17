use tracing::trace;

use crate::context::CodegenCx;
use crate::llvm::*;

// for some reason nvvm doesnt accept "irregular" types like i4 or i48, so we need to resort to using "regular"
// types and falling back to 1 if excrement hits the rotating fan.
const WIDTH_CANDIDATES: &[u32] = &[64, 32, 16, 8, 1];

/// returns Some(true) if the type is an integer of width 64, 32, 16, 8, or 1. Some(false) if its an int, and None if it
/// is not an int.
pub(crate) fn is_regular_int(ty: &Type) -> Option<bool> {
    unsafe {
        let kind = LLVMRustGetTypeKind(ty);
        if kind == TypeKind::Integer {
            let width = LLVMGetIntTypeWidth(ty);
            if WIDTH_CANDIDATES.contains(&width) {
                Some(true)
            } else {
                Some(false)
            }
        } else {
            None
        }
    }
}

/// Transforms a type to an nvvm-friendly vector type if the type is an int over i64.
/// returns a bool on whether the type was transformed.
pub(crate) fn get_transformed_type<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    ty: &'ll Type,
) -> (&'ll Type, bool) {
    unsafe {
        if is_regular_int(ty) == Some(false) {
            let width = LLVMGetIntTypeWidth(ty);
            let (width, count) = target_vector_width_and_count(width);
            let int_ty = LLVMIntTypeInContext(cx.llcx, width);
            trace!(
                "Transforming irregular int type `{:?}` to vector ty `{:?}` with length {}",
                ty,
                int_ty,
                count
            );
            (LLVMVectorType(int_ty, count), true)
        } else {
            (ty, false)
        }
    }
}

// try to find the largest possible int type to use for the target vector type.
// going from i64 down.
pub(crate) fn target_vector_width_and_count(int_width: u32) -> (u32, u32) {
    for &i in WIDTH_CANDIDATES {
        if int_width % i == 0 {
            return (i, int_width / i);
        }
    }
    unreachable!()
}
