use crate::LlvmMod;
use crate::llvm7;
use crate::llvm7::{False, True};
use crate::target;
use libc::c_uint;
use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorKind, AllocatorTy, alloc_error_handler_name, default_fn_name,
};
use rustc_middle::ty::TyCtxt;

// adapted from rustc_codegen_llvm
pub(crate) unsafe fn codegen(
    _tcx: TyCtxt<'_>,
    mods: &mut LlvmMod,
    _module_name: &str,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let llcx = &*mods.llcx;
    let llmod = unsafe { mods.llmod.as_ref().unwrap() };
    let usize = unsafe { target::usize_ty(llcx) };
    let i8 = unsafe { llvm7::LLVMInt8TypeInContext(llcx) };
    let i8p = unsafe { llvm7::LLVMPointerType(i8, 0) };
    let void = unsafe { llvm7::LLVMVoidTypeInContext(llcx) };

    let mut used = Vec::new();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut args = Vec::with_capacity(method.inputs.len());
            for ty in method.inputs.iter() {
                match ty.ty {
                    AllocatorTy::Layout => {
                        args.push(usize); // size
                        args.push(usize); // align
                    }
                    AllocatorTy::Ptr => args.push(i8p),
                    AllocatorTy::Usize => args.push(usize),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(i8p),
                AllocatorTy::Unit => None,

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let ty = unsafe {
                llvm7::LLVMFunctionType(
                    output.unwrap_or(void),
                    args.as_ptr(),
                    args.len() as c_uint,
                    False,
                )
            };
            let name = format!("__rust_{}", method.name);
            let llfn = unsafe {
                llvm7::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty)
            };

            used.push(llfn);
            // nvvm doesnt support uwtable so dont try to generate it

            let callee = default_fn_name(method.name);
            let callee = unsafe {
                llvm7::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty)
            };
            unsafe { llvm7::LLVMRustSetVisibility(callee, llvm7::Visibility::Hidden) };

            let llbb = unsafe {
                llvm7::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr().cast())
            };

            let llbuilder = unsafe { llvm7::LLVMCreateBuilderInContext(llcx) };
            unsafe { llvm7::LLVMPositionBuilderAtEnd(llbuilder, llbb) };
            let args = args
                .iter()
                .enumerate()
                .map(|(i, _)| unsafe { llvm7::LLVMGetParam(llfn, i as c_uint) })
                .collect::<Vec<_>>();
            // TODO: pass FnTy
            let ret = unsafe {
                llvm7::LLVMRustBuildCall(
                    llbuilder,
                    callee,
                    args.as_ptr(),
                    args.len() as c_uint,
                    None,
                )
            };
            unsafe { llvm7::LLVMSetTailCall(ret, True) };
            if output.is_some() {
                unsafe { llvm7::LLVMBuildRet(llbuilder, ret) };
            } else {
                unsafe { llvm7::LLVMBuildRetVoid(llbuilder) };
            }
            unsafe { llvm7::LLVMDisposeBuilder(llbuilder) };
        }
    }

    // rust alloc error handler
    let args = [usize, usize]; // size, align

    let ty = unsafe { llvm7::LLVMFunctionType(void, args.as_ptr(), args.len() as c_uint, False) };
    let name = "__rust_alloc_error_handler".to_string();
    let llfn =
        unsafe { llvm7::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty) };

    used.push(llfn);

    // -> ! DIFlagNoReturn
    llvm7::Attribute::NoReturn.apply_llfn(llvm7::AttributePlace::Function, llfn);

    let callee = alloc_error_handler_name(alloc_error_handler_kind);
    let callee = unsafe {
        llvm7::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty)
    };

    used.push(callee);

    // -> ! DIFlagNoReturn
    llvm7::Attribute::NoReturn.apply_llfn(llvm7::AttributePlace::Function, callee);
    unsafe { llvm7::LLVMRustSetVisibility(callee, llvm7::Visibility::Hidden) };

    let llbb = unsafe { llvm7::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr().cast()) };

    let llbuilder = unsafe { llvm7::LLVMCreateBuilderInContext(llcx) };
    unsafe { llvm7::LLVMPositionBuilderAtEnd(llbuilder, llbb) };
    let args = args
        .iter()
        .enumerate()
        .map(|(i, _)| unsafe { llvm7::LLVMGetParam(llfn, i as c_uint) })
        .collect::<Vec<_>>();
    // TODO: pass FnTy
    let ret = unsafe {
        llvm7::LLVMRustBuildCall(llbuilder, callee, args.as_ptr(), args.len() as c_uint, None)
    };
    unsafe { llvm7::LLVMSetTailCall(ret, True) };
    unsafe { llvm7::LLVMBuildRetVoid(llbuilder) };
    unsafe { llvm7::LLVMDisposeBuilder(llbuilder) };

    let ptr_ty = unsafe { llvm7::LLVMPointerType(llvm7::LLVMInt8TypeInContext(llcx), 0) };

    for used in &mut used {
        *used = unsafe { llvm7::LLVMConstBitCast(used, ptr_ty) };
    }

    let section = c"llvm.metadata";
    let array = unsafe { llvm7::LLVMConstArray(ptr_ty, used.as_ptr(), used.len() as u32) };
    let g = unsafe {
        llvm7::LLVMAddGlobal(llmod, llvm7::LLVMTypeOf(array), c"llvm.used".as_ptr().cast())
    };
    unsafe { llvm7::LLVMSetInitializer(g, array) };
    unsafe { llvm7::LLVMRustSetLinkage(g, llvm7::Linkage::AppendingLinkage) };
    unsafe { llvm7::LLVMSetSection(g, section.as_ptr()) };
}
