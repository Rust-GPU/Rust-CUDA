use tracing::trace;

use crate::builder::unnamed;
use crate::context::CodegenCx;
use crate::llvm;
use crate::llvm::{True, TypeKind, Type};

// for some reason nvvm doesnt accept "irregular" types like i4 or i48, so we need to resort to using "regular"
// types and falling back to 1 if excrement hits the rotating fan.
const WIDTH_CANDIDATES: &[u32] = &[64, 32, 16, 8, 1];

/// returns `true` if the type is or contains irregular integers.
pub(crate) fn type_needs_transformation(ty: &Type) -> bool {
    unsafe {
        let kind = llvm::LLVMRustGetTypeKind(ty);
        match kind {
            TypeKind::Integer => {
                let width = llvm::LLVMGetIntTypeWidth(ty);
                !WIDTH_CANDIDATES.contains(&width)
            }
            TypeKind::Struct => struct_type_fields(ty)
                .into_iter()
                .any(type_needs_transformation),
            _ => false,
        }
    }
}

fn struct_type_fields(ty: &Type) -> Vec<&Type> {
    unsafe {
        let count = llvm::LLVMCountStructElementTypes(ty);
        let mut fields = Vec::with_capacity(count as usize);
        llvm::LLVMGetStructElementTypes(ty, fields.as_mut_ptr());
        fields.set_len(count as usize);
        fields
    }
}

/// Transforms a type to an nvvm-friendly vector type if the type is an int over i64.
/// returns a bool on whether the type was transformed.
pub(crate) fn get_transformed_type<'ll>(
    cx: &CodegenCx<'ll, '_>,
    ty: &'ll Type,
) -> (&'ll Type, bool) {
    unsafe {
        if type_needs_transformation(ty) {
            let kind = llvm::LLVMRustGetTypeKind(ty);
            match kind {
                TypeKind::Integer => {
                    let width = llvm::LLVMGetIntTypeWidth(ty);
                    let (width, count) = target_vector_width_and_count(width);
                    let int_ty = llvm::LLVMIntTypeInContext(cx.llcx, width);
                    trace!(
                        "Transforming irregular int type `{:?}` to vector ty `{:?}` with length {}",
                        ty, int_ty, count
                    );
                    (llvm::LLVMVectorType(int_ty, count), true)
                }
                TypeKind::Struct => {
                    let fields = struct_type_fields(ty);
                    let transformed = fields
                        .into_iter()
                        .map(|field| get_transformed_type(cx, field).0)
                        .collect::<Vec<_>>();

                    let packed = llvm::LLVMIsPackedStruct(ty);
                    (cx.type_struct(&transformed, packed == True), true)
                }
                _ => unreachable!(),
            }
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

/// transmutes a value to a certain type, accounting for structs.
pub(crate) fn transmute_llval<'ll>(
    bx: &mut llvm::Builder<'ll>,
    _cx: &CodegenCx<'ll, '_>,
    a_val: &'ll llvm::Value,
    ty: &'ll llvm::Type,
) -> &'ll llvm::Value {
    trace!("transmute_llval: transmuting `{:?}` to `{:?}`", a_val, ty);
    unsafe {
        let kind = llvm::LLVMRustGetTypeKind(ty);
        match kind {
            // structs cannot be bitcasted, so we need to do it using a bunch of extract/insert values.
            TypeKind::Struct => {
                let new_struct = llvm::LLVMGetUndef(ty);
                let mut last_val = new_struct;
                for (idx, field) in struct_type_fields(ty).into_iter().enumerate() {
                    let field_val = llvm::LLVMBuildExtractValue(bx, a_val, idx as u32, unnamed());
                    let new_val = transmute_llval(bx, _cx, field_val, field);
                    last_val = llvm::LLVMBuildInsertValue(bx, last_val, new_val, idx as u32, unnamed());
                }
                last_val
            }
            _ => llvm::LLVMBuildBitCast(bx, a_val, ty, unnamed()),
        }
    }
}
