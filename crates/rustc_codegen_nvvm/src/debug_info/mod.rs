use std::cell::RefCell;
use std::ffi::CString;
use std::ops::Range;

use libc::c_uint;
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_codegen_ssa::mir::debuginfo::VariableKind::*;
use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext, VariableKind};
use rustc_codegen_ssa::traits::DebugInfoBuilderMethods;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_index::vec::IndexVec;
use rustc_middle::mir;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{
    self, Binder, ExistentialTraitRef, Instance, ParamEnv, PolyExistentialTraitRef, Ty,
    TypeFoldable, TypeVisitable,
};
use rustc_session::config::{self, DebugInfo};
use rustc_span::symbol::Symbol;
use rustc_span::{self, BytePos, Pos, SourceFile, SourceFileAndLine, Span};
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::{Primitive, Size};

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::debug_info::util::{create_DIArray, is_node_local_to_unit};
use crate::llvm::{self, debuginfo::*, Value};

use self::namespace::*;
use self::util::DIB;
use create_scope_map::compute_mir_scopes;

mod create_scope_map;
mod metadata;
mod namespace;
mod util;

pub(crate) use metadata::*;

#[allow(non_upper_case_globals)]
const DW_TAG_auto_variable: c_uint = 0x100;
#[allow(non_upper_case_globals)]
const DW_TAG_arg_variable: c_uint = 0x101;

pub struct CrateDebugContext<'a, 'tcx> {
    #[allow(dead_code)]
    llcontext: &'a llvm::Context,
    llmod: &'a llvm::Module,
    builder: &'a mut DIBuilder<'a>,
    #[allow(clippy::type_complexity)]
    created_files: RefCell<FxHashMap<(Option<String>, Option<String>), &'a DIFile>>,
    created_enum_disr_types: RefCell<FxHashMap<(DefId, Primitive), &'a DIType>>,

    type_map: RefCell<TypeMap<'a, 'tcx>>,
    namespace_map: RefCell<DefIdMap<&'a DIScope>>,

    // This collection is used to assert that composite types (structs, enums,
    // ...) have their members only set once:
    composite_types_completed: RefCell<FxHashSet<&'a DIType>>,
}

impl<'a, 'tcx> Drop for CrateDebugContext<'a, 'tcx> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustDIBuilderDispose(&mut *(self.builder as *mut _));
        }
    }
}

impl<'a, 'tcx> CrateDebugContext<'a, 'tcx> {
    pub(crate) fn new(llmod: &'a llvm::Module) -> Self {
        let builder = unsafe { llvm::LLVMRustDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        CrateDebugContext {
            llcontext,
            llmod,
            builder,
            created_files: Default::default(),
            created_enum_disr_types: Default::default(),
            type_map: Default::default(),
            namespace_map: RefCell::new(Default::default()),
            composite_types_completed: Default::default(),
        }
    }

    pub fn finalize(&self) {
        unsafe {
            llvm::LLVMRustDIBuilderFinalize(self.builder);

            // Prevent bitcode readers from deleting the debug info.
            let ptr = "Debug Info Version\0".as_ptr();
            llvm::LLVMRustAddModuleFlag(
                self.llmod,
                ptr.cast(),
                llvm::LLVMRustDebugMetadataVersion(),
            );
        }
    }
}

impl<'a, 'll, 'tcx> DebugInfoBuilderMethods for Builder<'a, 'll, 'tcx> {
    fn dbg_var_addr(
        &mut self,
        dbg_var: &'ll DIVariable,
        dbg_loc: &'ll DILocation,
        variable_alloca: &'ll Value,
        direct_offset: Size,
        indirect_offsets: &[Size],
        fragment: Option<Range<Size>>,
    ) {
        let op_deref = || unsafe { llvm::LLVMRustDIBuilderCreateOpDeref() };
        let op_plus_uconst = || unsafe { llvm::LLVMRustDIBuilderCreateOpPlusUconst() };
        let mut addr_ops = Vec::new();

        if direct_offset.bytes() > 0 {
            addr_ops.push(op_plus_uconst());
            addr_ops.push(direct_offset.bytes() as i64);
        }
        for &offset in indirect_offsets {
            addr_ops.push(op_deref());
            if offset.bytes() > 0 {
                addr_ops.push(op_plus_uconst());
                addr_ops.push(offset.bytes() as i64);
            }
        }

        unsafe {
            llvm::LLVMRustDIBuilderInsertDeclareAtEnd(
                DIB(self.cx()),
                variable_alloca,
                dbg_var,
                addr_ops.as_ptr(),
                addr_ops.len() as c_uint,
                dbg_loc,
                self.llbb(),
            );
        }
    }

    fn set_dbg_loc(&mut self, dbg_loc: &'ll DILocation) {
        unsafe {
            let dbg_loc_as_llval = llvm::LLVMRustMetadataAsValue(self.cx().llcx, dbg_loc);
            llvm::LLVMSetCurrentDebugLocation(self.llbuilder, dbg_loc_as_llval);
        }
    }

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        // do nothing
    }

    fn set_var_name(&mut self, value: &'ll Value, name: &str) {
        if self.sess().fewer_names() {
            return;
        }

        // Only function parameters and instructions are local to a function,
        // don't change the name of anything else (e.g. globals).
        let param_or_inst = unsafe {
            llvm::LLVMIsAArgument(value).is_some() || llvm::LLVMIsAInstruction(value).is_some()
        };
        if !param_or_inst {
            return;
        }

        // Avoid replacing the name if it already exists.
        // While we could combine the names somehow, it'd
        // get noisy quick, and the usefulness is dubious.
        if llvm::get_value_name(value).is_empty() {
            llvm::set_value_name(value, name.as_bytes());
        }
    }
}

/// A source code location used to generate debug information.
pub struct DebugLoc {
    /// Information about the original source file.
    pub file: Lrc<SourceFile>,
    /// The (1-based) line number.
    pub line: u32,
    /// The (1-based) column number.
    pub col: u32,
}

impl<'ll> CodegenCx<'ll, '_> {
    /// Looks up debug source information about a `BytePos`.
    pub fn lookup_debug_loc(&self, pos: BytePos) -> DebugLoc {
        let (file, line, col) = match self.sess().source_map().lookup_line(pos) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.line_begin_pos(pos);

                // Use 1-based indexing.
                let line = (line + 1) as u32;
                let col = (pos - line_pos).to_u32() + 1;

                (file, line, col)
            }
            Err(file) => (file, UNKNOWN_LINE_NUMBER, UNKNOWN_COLUMN_NUMBER),
        };

        DebugLoc { file, line, col }
    }
}

impl<'ll, 'tcx> DebugInfoMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: &'ll Value,
        mir: &mir::Body<'tcx>,
    ) -> Option<FunctionDebugContext<&'ll DIScope, &'ll DILocation>> {
        if self.sess().opts.debuginfo == DebugInfo::None {
            return None;
        }

        // Initialize fn debug context (including scopes).
        let empty_scope = DebugScope {
            dbg_scope: self.dbg_scope_fn(instance, fn_abi, Some(llfn)),
            inlined_at: None,
            file_start_pos: BytePos(0),
            file_end_pos: BytePos(0),
        };
        let mut fn_debug_context = FunctionDebugContext {
            scopes: IndexVec::from_elem(empty_scope, &mir.source_scopes),
        };

        // Fill in all the scopes, with the information from the MIR body.
        compute_mir_scopes(self, instance, mir, &mut fn_debug_context);

        Some(fn_debug_context)
    }

    fn create_vtable_debuginfo(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<PolyExistentialTraitRef<'tcx>>,
        vtable: &'ll Value,
    ) {
        // todo!();
    }

    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<&'ll Value>,
    ) -> &'ll DIScope {
        let def_id = instance.def_id();
        let containing_scope = get_containing_scope(self, instance);
        let span = self.tcx.def_span(def_id);
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let function_type_metadata = unsafe {
            let fn_signature = get_function_signature(self, fn_abi);
            llvm::LLVMRustDIBuilderCreateSubroutineType(DIB(self), fn_signature)
        };

        let mut name = String::new();
        type_names::push_item_name(self.tcx(), def_id, false, &mut name);

        // Find the enclosing function, in case this is a closure.
        let enclosing_fn_def_id = self.tcx().typeck_root_def_id(def_id);

        // Get_template_parameters() will append a `<...>` clause to the function
        // name if necessary.
        let generics = self.tcx().generics_of(enclosing_fn_def_id);
        let substs = instance.substs.truncate_to(self.tcx(), generics);
        let template_parameters = get_template_parameters(self, generics, substs, &mut name);

        let linkage_name = &mangled_name_of_instance(self, instance).name;
        // Omit the linkage_name if it is the same as subprogram name.
        let linkage_name = if &name == linkage_name {
            ""
        } else {
            linkage_name
        };
        let name = CString::new(name).unwrap();
        let linkage_name = CString::new(linkage_name).unwrap();

        let scope_line = loc.line;

        let mut flags = DIFlags::FlagPrototyped;

        if fn_abi.ret.layout.abi.is_uninhabited() {
            flags |= DIFlags::FlagNoReturn;
        }

        unsafe {
            return llvm::LLVMRustDIBuilderCreateFunction(
                DIB(self),
                containing_scope,
                name.as_ptr(),
                linkage_name.as_ptr(),
                file_metadata,
                loc.line,
                function_type_metadata,
                is_node_local_to_unit(self, def_id),
                true,
                scope_line,
                flags,
                self.sess().opts.optimize != config::OptLevel::No,
                maybe_definition_llfn,
                template_parameters,
                None,
            );
        }

        fn get_function_signature<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        ) -> &'ll DIArray {
            if cx.sess().opts.debuginfo == DebugInfo::Limited {
                return create_DIArray(DIB(cx), &[]);
            }

            let mut signature = Vec::with_capacity(fn_abi.args.len() + 1);

            // Return type -- llvm::DIBuilder wants this at index 0
            signature.push(if fn_abi.ret.is_ignore() {
                None
            } else {
                Some(type_metadata(
                    cx,
                    fn_abi.ret.layout.ty,
                    rustc_span::DUMMY_SP,
                ))
            });

            signature.extend(
                fn_abi
                    .args
                    .iter()
                    .map(|arg| Some(type_metadata(cx, arg.layout.ty, rustc_span::DUMMY_SP))),
            );

            create_DIArray(DIB(cx), &signature[..])
        }

        fn get_template_parameters<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            generics: &ty::Generics,
            substs: SubstsRef<'tcx>,
            name_to_append_suffix_to: &mut String,
        ) -> &'ll DIArray {
            type_names::push_generic_params(
                cx.tcx,
                cx.tcx
                    .normalize_erasing_regions(ty::ParamEnv::reveal_all(), substs),
                name_to_append_suffix_to,
            );

            if substs.types().next().is_none() {
                return create_DIArray(DIB(cx), &[]);
            }

            // Again, only create type information if full debuginfo is enabled
            let template_params: Vec<_> = if cx.sess().opts.debuginfo == DebugInfo::Full {
                let names = get_parameter_names(cx, generics);
                substs
                    .into_iter()
                    .zip(names.into_iter())
                    .filter_map(|(kind, name)| {
                        if let GenericArgKind::Type(ty) = kind.unpack() {
                            let actual_type =
                                cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);
                            let actual_type_metadata =
                                type_metadata(cx, actual_type, rustc_span::DUMMY_SP);
                            let name = CString::new(&*name.as_str()).unwrap();
                            Some(unsafe {
                                Some(llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                                    DIB(cx),
                                    None,
                                    name.as_ptr().cast(),
                                    actual_type_metadata,
                                ))
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            };

            create_DIArray(DIB(cx), &template_params[..])
        }

        fn get_parameter_names(cx: &CodegenCx<'_, '_>, generics: &ty::Generics) -> Vec<Symbol> {
            let mut names = generics.parent.map_or_else(Vec::new, |def_id| {
                get_parameter_names(cx, cx.tcx.generics_of(def_id))
            });
            names.extend(generics.params.iter().map(|param| param.name));
            names
        }

        fn get_containing_scope<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            instance: Instance<'tcx>,
        ) -> &'ll DIScope {
            // First, let's see if this is a method within an inherent impl. Because
            // if yes, we want to make the result subroutine DIE a child of the
            // subroutine's self-type.
            let self_type = cx
                .tcx
                .impl_of_method(instance.def_id())
                .and_then(|impl_def_id| {
                    // If the method does *not* belong to a trait, proceed
                    if cx.tcx.trait_id_of_impl(impl_def_id).is_none() {
                        let impl_self_ty = cx.tcx.subst_and_normalize_erasing_regions(
                            instance.substs,
                            ty::ParamEnv::reveal_all(),
                            cx.tcx.type_of(impl_def_id),
                        );

                        // Only "class" methods are generally understood by LLVM,
                        // so avoid methods on other types (e.g., `<*mut T>::null`).
                        match impl_self_ty.kind() {
                            ty::Adt(def, ..) if !def.is_box() => {
                                // Again, only create type information if full debuginfo is enabled
                                if cx.sess().opts.debuginfo == DebugInfo::Full
                                    && !impl_self_ty.needs_subst()
                                {
                                    Some(type_metadata(cx, impl_self_ty, rustc_span::DUMMY_SP))
                                } else {
                                    Some(namespace::item_namespace(cx, def.did()))
                                }
                            }
                            _ => None,
                        }
                    } else {
                        // For trait method impls we still use the "parallel namespace"
                        // strategy
                        None
                    }
                });

            self_type.unwrap_or_else(|| {
                namespace::item_namespace(
                    cx,
                    DefId {
                        krate: instance.def_id().krate,
                        index: cx
                            .tcx
                            .def_key(instance.def_id())
                            .parent
                            .expect("get_containing_scope: missing parent?"),
                    },
                )
            })
        }
    }

    fn dbg_loc(
        &self,
        scope: Self::DIScope,
        inlined_at: Option<Self::DILocation>,
        span: Span,
    ) -> Self::DILocation {
        let DebugLoc { line, col, .. } = self.lookup_debug_loc(span.lo());

        unsafe { llvm::LLVMRustDIBuilderCreateDebugLocation(line, col, scope, inlined_at) }
    }

    // fn create_vtable_metadata(
    //     &self,
    //     ty: Ty<'tcx>,
    //     _: Option<Binder<'tcx, ExistentialTraitRef<'tcx>>>,
    //     vtable: Self::Value,
    // ) {
    //     metadata::create_vtable_metadata(self, ty, vtable)
    // }

    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
    ) -> Self::DIScope {
        metadata::extend_scope_to_file(self, scope_metadata, file)
    }

    fn debuginfo_finalize(&self) {
        if let Some(dbg_cx) = &self.dbg_cx {
            dbg_cx.finalize();
        }
    }

    fn create_dbg_var(
        &self,
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_kind: VariableKind,
        span: Span,
    ) -> Self::DIVariable {
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let type_metadata = type_metadata(self, variable_type, span);

        let (argument_index, dwarf_tag) = match variable_kind {
            ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
            LocalVariable => (0, DW_TAG_auto_variable),
        };
        let align = self.align_of(variable_type);

        let name = CString::new(&*variable_name.as_str()).unwrap();
        unsafe {
            llvm::LLVMRustDIBuilderCreateVariable(
                DIB(self),
                dwarf_tag,
                scope_metadata,
                name.as_ptr().cast(),
                file_metadata,
                loc.line,
                type_metadata,
                true,
                DIFlags::FlagZero,
                argument_index,
                align.bytes() as u32,
            )
        }
    }
}
