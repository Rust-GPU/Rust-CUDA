use std::io::{self, Write};
use std::slice;
use std::sync::Arc;
use std::ffi::CString;

use libc::{c_char, size_t};
use rustc_codegen_ssa::back::write::{TargetMachineFactoryConfig, TargetMachineFactoryFn};
use rustc_codegen_ssa::traits::{DebugInfoCodegenMethods, MiscCodegenMethods};
use rustc_codegen_ssa::{
    CompiledModule, ModuleCodegen,
    back::write::{CodegenContext, ModuleConfig},
    base::maybe_create_entry_wrapper,
    mono_item::MonoItemExt,
    traits::{BaseTypeCodegenMethods, ThinBufferMethods},
};
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_middle::mir::mono::{MonoItem, MonoItemData};
use rustc_middle::{
    dep_graph, 
    ty::TyCtxt
};
use rustc_session::Session;
use rustc_session::config::{self, DebugInfo, OutputType};
use rustc_span::Symbol;
use rustc_target::spec::{CodeModel, RelocModel};

use crate::common::AsCCharPtr;
use crate::llvm;
use crate::override_fns::define_or_override_fn;
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::lto::ThinBuffer;
use crate::LlvmMod;
use crate::NvvmCodegenBackend;

pub fn llvm_err(handle: DiagCtxtHandle, msg: &str) -> FatalError {
    match llvm::last_error() {
        Some(err) => handle.fatal(format!("{}: {}", msg, err)),
        None => handle.fatal(msg.to_string()),
    }
}

pub fn to_llvm_opt_settings(
    cfg: config::OptLevel,
) -> (llvm::CodeGenOptLevel, llvm::CodeGenOptSize) {
    use self::config::OptLevel::*;
    match cfg {
        No => (llvm::CodeGenOptLevel::None, llvm::CodeGenOptSizeNone),
        Less => (llvm::CodeGenOptLevel::Less, llvm::CodeGenOptSizeNone),
        More => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeNone),
        Aggressive => (llvm::CodeGenOptLevel::Aggressive, llvm::CodeGenOptSizeNone),
        Size => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeDefault),
        SizeMin => (
            llvm::CodeGenOptLevel::Default,
            llvm::CodeGenOptSizeAggressive,
        ),
    }
}

fn to_llvm_relocation_model(relocation_model: RelocModel) -> llvm::RelocMode {
    match relocation_model {
        RelocModel::Static => llvm::RelocMode::Static,
        RelocModel::Pic => llvm::RelocMode::PIC,
        RelocModel::DynamicNoPic => llvm::RelocMode::DynamicNoPic,
        RelocModel::Ropi => llvm::RelocMode::ROPI,
        RelocModel::Rwpi => llvm::RelocMode::RWPI,
        RelocModel::RopiRwpi => llvm::RelocMode::ROPI_RWPI,
        RelocModel::Pie => panic!(),
    }
}

pub(crate) fn to_llvm_code_model(code_model: Option<CodeModel>) -> llvm::CodeModel {
    match code_model {
        Some(CodeModel::Tiny) => llvm::CodeModel::Small,
        Some(CodeModel::Small) => llvm::CodeModel::Small,
        Some(CodeModel::Kernel) => llvm::CodeModel::Kernel,
        Some(CodeModel::Medium) => llvm::CodeModel::Medium,
        Some(CodeModel::Large) => llvm::CodeModel::Large,
        None => llvm::CodeModel::None,
    }
}

pub fn target_machine_factory(
    sess: &Session,
    optlvl: config::OptLevel,
) -> TargetMachineFactoryFn<NvvmCodegenBackend> {
    let reloc_model = to_llvm_relocation_model(sess.relocation_model());

    let (opt_level, _) = to_llvm_opt_settings(optlvl);
    let use_softfp = sess.opts.cg.soft_float;

    let ffunction_sections = sess
        .opts
        .unstable_opts
        .function_sections
        .unwrap_or(sess.target.function_sections);
    let fdata_sections = ffunction_sections;

    let code_model = to_llvm_code_model(sess.code_model());

    let triple = sess.target.llvm_target.clone().to_string();
    let cpu_string = sess.opts.cg.target_cpu
        .as_deref()
        .unwrap_or("") // TODO: sm_120?
        .to_string();
    let features_string = "".to_string();
    let trap_unreachable = sess
        .opts
        .unstable_opts
        .trap_unreachable
        .unwrap_or(sess.target.trap_unreachable);

    Arc::new(move |_config: TargetMachineFactoryConfig| {
        let triple_cstr = CString::new(triple.as_str())
            .map_err(|e| format!("Invalid triple string: {}", e))?;
        let cpu_cstr = CString::new(cpu_string.as_str())
            .map_err(|e| format!("Invalid CPU string: {}", e))?;
        let features_cstr = CString::new(features_string.as_str())
            .map_err(|e| format!("Invalid features string: {}", e))?;
        let abi_cstr = CString::new("").unwrap();
        let debug_compression_cstr = CString::new("none").unwrap();

        let tm = unsafe {
            llvm::LLVMRustCreateTargetMachine(
                triple_cstr.as_ptr(), // triple
                cpu_cstr.as_ptr(), // cpu
                features_cstr.as_ptr(), // feature
                abi_cstr.as_ptr(), // abistr
                code_model, // RustCM
                reloc_model, // reloc
                opt_level, // opt
                if use_softfp { llvm::FloatABIType::Soft } else { llvm::FloatABIType::Default }, // FloatABIType
                ffunction_sections, // function sections
                fdata_sections,
                false, // unique section names
                trap_unreachable,
                false, // single thread
                false, // VerboseAsm: bool,
                false, // EmitStackSizeSection: bool,
                false, // RelaxELFRelocations: bool,
                false, // UseInitArray: bool,
                std::ptr::null(), // SplitDwarfFile: *const c_char,
                std::ptr::null(), // OutputObjFile: *const c_char,
                debug_compression_cstr.as_ptr(), // DebugInfoCompression: *const c_char,
                false, // UseEmulatedTls: bool,
                std::ptr::null(), // ArgsCstrBuff: *const c_char,
                0, // ArgsCstrBuffLen: usize,
            )
        };
        tm.ok_or_else(|| format!("Could not create LLVM TargetMachine for triple: {}", triple))
    })
}

pub extern "C" fn demangle_callback(
    input_ptr: *const c_char,
    input_len: size_t,
    output_ptr: *mut c_char,
    output_len: size_t,
) -> size_t {
    let input = unsafe { slice::from_raw_parts(input_ptr as *const u8, input_len) };

    let input = match std::str::from_utf8(input) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let output = unsafe { slice::from_raw_parts_mut(output_ptr as *mut u8, output_len) };
    let mut cursor = io::Cursor::new(output);

    let demangled = match rustc_demangle::try_demangle(input) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    if write!(cursor, "{:#}", demangled).is_err() {
        // Possible only if provided buffer is not big enough
        return 0;
    }

    cursor.position() as size_t
}

/// Compile a single module (in an nvvm context this means getting the llvm bitcode out of it)
pub(crate) unsafe fn codegen(
    cgcx: &CodegenContext<NvvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<LlvmMod>,
    config: &ModuleConfig,
) -> Result<CompiledModule, FatalError> {
    // For NVVM, all the codegen we need to do is turn the llvm modules
    // into llvm bitcode and write them to a tempdir. nvvm expects llvm
    // bitcode as the modules to be added to the program. Then as the last step
    // we gather all those tasty bitcode files, add them to the nvvm program
    // and finally tell nvvm to compile it, which gives us a ptx file.
    //
    // we also implement emit_ir so we can dump the IR fed to nvvm in case we
    // feed it anything it doesnt like

    let _timer = cgcx
        .prof
        .generic_activity_with_arg("NVVM_module_codegen", &module.name[..]);

    let llmod = unsafe { module.module_llvm.llmod.as_ref().unwrap() };
    let mod_name = module.name.clone();
    let module_name = Some(&mod_name[..]);

    let out = cgcx
        .output_filenames
        .temp_path(OutputType::Object, module_name);

    // nvvm ir *is* llvm ir so emit_ir fits the expectation of llvm ir which is why we
    // implement this. this is copy and pasted straight from rustc_codegen_llvm
    // because im too lazy to make it seem like i rewrote this when its the same logic
    if config.emit_ir {
        let _timer = cgcx
            .prof
            .generic_activity_with_arg("NVVM_module_codegen_emit_ir", &module.name[..]);
        let out = cgcx
            .output_filenames
            .temp_path(OutputType::LlvmAssembly, module_name);
        let out = out.to_str().unwrap();

        let result = unsafe {
            llvm::LLVMRustPrintModule(llmod, out.as_c_char_ptr(), out.len(), demangle_callback)
        };

        result.into_result().map_err(|()| {
            let msg = format!("failed to write NVVM IR to {}", out);
            llvm_err(dcx, &msg)
        })?;
    }

    let _bc_timer = cgcx
        .prof
        .generic_activity_with_arg("NVVM_module_codegen_make_bitcode", &module.name[..]);

    let thin = ThinBuffer::new(llmod);

    let data = thin.data();

    let _bc_emit_timer = cgcx
        .prof
        .generic_activity_with_arg("NVVM_module_codegen_emit_bitcode", &module.name[..]);

    if let Err(e) = std::fs::write(&out, data) {
        let msg = format!("failed to write bytecode to {}: {}", out.display(), e);
        dcx.err(msg);
    }

    Ok(CompiledModule {
        name: mod_name,
        kind: module.kind,
        object: Some(out),
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
    })
}

/// compile a single codegen unit.
/// This involves getting its llvm module and doing some housekeeping such as
/// monomorphizing items and using RAUW on statics. This codegenned module is then
/// given to other functions to "compile it" (in our case not really because nvvm does
/// codegen on all the modules at once) and then link it (once again, nvvm does linking and codegen
/// in a single step)
pub fn compile_codegen_unit(tcx: TyCtxt<'_>, cgu_name: Symbol) -> (ModuleCodegen<LlvmMod>, u64) {
    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) = tcx.dep_graph.with_task(
        dep_node,
        tcx,
        cgu_name,
        module_codegen,
        Some(dep_graph::hash_result),
    );

    fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<LlvmMod> {
        let cgu = tcx.codegen_unit(cgu_name);

        // Instantiate monomorphizations without filling out definitions yet...
        let llvm_module = LlvmMod::new(cgu_name.as_str());
        {
            let cx = CodegenCx::new(tcx, cgu, &llvm_module);

            let mono_items = cx.codegen_unit.items_in_deterministic_order(cx.tcx);

            for &(
                mono_item,
                MonoItemData {
                    linkage,
                    visibility,
                    ..
                },
            ) in &mono_items
            {
                mono_item.predefine::<Builder<'_, '_, '_>>(&cx, linkage, visibility);
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(mono_item, _) in &mono_items {
                if let MonoItem::Fn(func) = mono_item {
                    define_or_override_fn(func, &cx);
                } else {
                    mono_item.define::<Builder<'_, '_, '_>>(&cx);
                }
            }

            // a main function for gpu kernels really makes no sense but
            // codegen it anyways.
            // sanitize attrs are not allowed in nvvm so do nothing further.
            maybe_create_entry_wrapper::<Builder<'_, '_, '_>>(&cx);

            // Run replace-all-uses-with for statics that need it
            for &(old_g, new_g) in cx.statics_to_rauw.borrow().iter() {
                unsafe {
                    let bitcast = llvm::LLVMConstPointerCast(new_g, cx.val_ty(old_g));
                    llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                    llvm::LLVMDeleteGlobal(old_g);
                }
            }

            // Create the llvm.used and llvm.compiler.used variables.
            if !cx.used_statics.borrow().is_empty() {
                cx.create_used_variable_impl(c"llvm.used", &cx.used_statics.borrow());
            }
            if !cx.compiler_used_statics.borrow().is_empty() {
                cx.create_used_variable_impl(
                    c"llvm.compiler.used",
                    &cx.compiler_used_statics.borrow(),
                );
            }

            // Finalize debuginfo
            if cx.sess().opts.debuginfo != DebugInfo::None {
                cx.debuginfo_finalize();
            }
        }

        ModuleCodegen::new_regular(cgu_name.to_string(), llvm_module)
    }

    // TODO(RDambrosio016): maybe the same cost as the llvm codegen works?
    // nvvm does some exotic things and does linking too so it might be inaccurate
    (module, 0)
}

/*pub(crate) unsafe fn optimize(
    _cgcx: &CodegenContext<NvvmCodegenBackend>,
    _diag_handler: DiagCtxtHandle<'_>,
    _module: &ModuleCodegen<LlvmMod>,
    _config: &ModuleConfig,
) -> Result<(), FatalError> {
    // TODO: implement this
    Ok(())
}*/

// TODO: We use rustc's optimization approach from when it used llvm 7, because many things
// are incompatible with llvm 7 nowadays. Although we should probably consult a rustc dev on whether
// any big things were discovered in that timespan that we should modify.
pub(crate) unsafe fn optimize(
    cgcx: &CodegenContext<NvvmCodegenBackend>,
    diag_handler: DiagCtxtHandle<'_>,
    module: &ModuleCodegen<LlvmMod>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    
    let _timer = cgcx
        .prof
        .generic_activity_with_arg("LLVM_module_optimize", &module.name[..]);

    let llmod = unsafe { &*module.module_llvm.llmod };

    if config.emit_no_opt_bc {
        let mod_name = module.name.clone();
        let module_name = Some(&mod_name[..]);
        let out = cgcx
            .output_filenames
            .temp_path_ext("no-opt.bc", module_name);
        let out = rustc_fs_util::path_to_c_string(&out);
        unsafe { llvm::LLVMWriteBitcodeToFile(llmod, out.as_ptr()); }
    }

    let tm_factory_config = TargetMachineFactoryConfig {
        split_dwarf_file: None,
        output_obj_file: None,
    };

    let tm = (cgcx.tm_factory)(tm_factory_config).expect("failed to create target machine");

    // LLVM 19: Complete rewrite using new pass manager
    if config.opt_level.is_some() {
        unsafe {
            let mut error_msg = std::ptr::null_mut();
            let verify_result = llvm::LLVMVerifyModule(
                llmod,
                llvm::LLVMVerifierFailureAction::LLVMPrintMessageAction,
                &mut error_msg
            );
            if verify_result != 0 {
                llvm::LLVMDumpModule(llmod);
                rustc_middle::bug!("Module verification failed!");
            }

            // Create pass builder options
            let pass_options = llvm::LLVMCreatePassBuilderOptions();
            
            // Configure pass builder options based on config
            let opt_level = config
                .opt_level
                .map_or(llvm::CodeGenOptLevel::None, |x| to_llvm_opt_settings(x).0);
            
            // Set various options on the pass builder
            // TODO: support these flags
            /*if config.verify_each {
                llvm::LLVMPassBuilderOptionsSetVerifyEach(pass_options, 1);
            }
            
            // Enable debug logging if needed
            if config.debug_pass_manager {
                llvm::LLVMPassBuilderOptionsSetDebugLogging(pass_options, 1);
            }*/
            
            // Build the pass pipeline string based on optimization level and config
            let mut pass_pipeline = String::new();
            
            if !config.no_prepopulate_passes {
                // Use default optimization pipeline based on level
                match opt_level {
                    llvm::CodeGenOptLevel::None => {
                        pass_pipeline.push_str("default<O0>");
                    },
                    llvm::CodeGenOptLevel::Less => {
                        pass_pipeline.push_str("default<O1>");
                    },
                    llvm::CodeGenOptLevel::Default => {
                        pass_pipeline.push_str("default<O2>");
                    },
                    llvm::CodeGenOptLevel::Aggressive => {
                        pass_pipeline.push_str("default<O3>");
                    },
                }
            }
            
            // Add custom passes from config
            for pass in &config.passes {
                if !pass_pipeline.is_empty() {
                    pass_pipeline.push(',');
                }
                pass_pipeline.push_str(pass);
            }
            
            
            // Convert pass pipeline string to C string
            let c_pass_pipeline = std::ffi::CString::new(pass_pipeline)
                .expect("Pass pipeline string contains null byte");
            
            // Run the passes using the new pass manager
            
            let result = llvm::LLVMRunPasses(
                llmod,                          // Module
                c_pass_pipeline.as_ptr(),       // Pass pipeline string
                tm,                             // TargetMachine
                pass_options                    // PassBuilderOptions
            );
            
            
            if result != 0 {
                diag_handler.err("Failed to run optimization passes");
            } else {
            }
            
            // Clean up
            llvm::LLVMDisposePassBuilderOptions(pass_options);
        }
    } else {
    }

    Ok(())
}

// TODO: remove this dead code?
/*unsafe fn with_llvm_pmb(
    llmod: &llvm::Module,
    config: &ModuleConfig,
    opt_level: llvm::CodeGenOptLevel,
    f: &mut impl FnMut(&llvm::PassManagerBuilder),
) {
    unsafe {
        use std::ptr;

        let builder = llvm::LLVMPassManagerBuilderCreate();
        let opt_size = config
            .opt_size
            .map_or(llvm::CodeGenOptSizeNone, |x| to_llvm_opt_settings(x).1);

        llvm::LLVMRustConfigurePassManagerBuilder(
            builder,
            opt_level,
            config.merge_functions,
            config.vectorize_slp,
            config.vectorize_loop,
            false,
            ptr::null(),
            ptr::null(),
        );

        llvm::LLVMPassManagerBuilderSetSizeLevel(builder, opt_size as u32);

        if opt_size != llvm::CodeGenOptSizeNone {
            llvm::LLVMPassManagerBuilderSetDisableUnrollLoops(builder, 1);
        }

        llvm::LLVMRustAddBuilderLibraryInfo(builder, llmod, config.no_builtins);

        // Here we match what clang does (kinda). For O0 we only inline
        // always-inline functions (but don't add lifetime intrinsics), at O1 we
        // inline with lifetime intrinsics, and O2+ we add an inliner with a
        // thresholds copied from clang.
        match (opt_level, opt_size) {
            (llvm::CodeGenOptLevel::Aggressive, ..) => {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 275);
            }
            (_, llvm::CodeGenOptSizeDefault) => {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 75);
            }
            (_, llvm::CodeGenOptSizeAggressive) => {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 25);
            }
            (llvm::CodeGenOptLevel::None, ..) => {
                llvm::LLVMRustAddAlwaysInlinePass(builder, false);
            }
            (llvm::CodeGenOptLevel::Less, ..) => {
                llvm::LLVMRustAddAlwaysInlinePass(builder, true);
            }
            (llvm::CodeGenOptLevel::Default, ..) => {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 225);
            }
            (llvm::CodeGenOptLevel::Other, ..) => {
                bug!("CodeGenOptLevel::Other selected")
            }
        }

        f(builder);
        llvm::LLVMPassManagerBuilderDispose(builder);
    }
}*/
