use rustc_codegen_ssa::CodegenResults;
use rustc_codegen_ssa::CompiledModule;
use rustc_codegen_ssa::NativeLib;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::rustc_erase_owner;
use rustc_data_structures::sync::MetadataRef;
use rustc_hash::FxHashSet;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_session::cstore::MetadataLoader;
use rustc_session::output::out_filename;
use rustc_session::{
    config::{CrateType, OutputFilenames, OutputType},
    output::check_file_is_writeable,
    utils::NativeLibKind,
    Session,
};
use rustc_target::spec::Target;
use std::{
    ffi::OsStr,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
};
use tar::{Archive, Builder, Header};
use tracing::{debug, trace};

use crate::context::CodegenArgs;
use crate::LlvmMod;
use rustc_ast::CRATE_NODE_ID;

pub(crate) struct NvvmMetadataLoader;

impl MetadataLoader for NvvmMetadataLoader {
    fn get_rlib_metadata(&self, _target: &Target, filename: &Path) -> Result<MetadataRef, String> {
        trace!("Retrieving rlib metadata for `{:?}`", filename);
        read_metadata(filename)
    }

    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String> {
        trace!("Retrieving dylib metadata for `{:?}`", filename);
        // This is required for loading metadata from proc macro crates compiled as dylibs for the host target.
        rustc_codegen_llvm::LlvmCodegenBackend::new()
            .metadata_loader()
            .get_dylib_metadata(target, filename)
    }
}

fn read_metadata(rlib: &Path) -> Result<MetadataRef, String> {
    let read_meta = || -> Result<Option<MetadataRef>, io::Error> {
        for entry in Archive::new(File::open(rlib)?).entries()? {
            let mut entry = entry?;
            if entry.path()? == Path::new(".metadata") {
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(bytes);
                return Ok(Some(rustc_erase_owner!(buf.map_owner_box())));
            }
        }
        Ok(None)
    };

    match read_meta() {
        Ok(Some(m)) => Ok(m),
        Ok(None) => Err(format!("No .metadata file in rlib: {:?}", rlib)),
        Err(io) => Err(format!("Failed to read rlib at {:?}: {}", rlib, io)),
    }
}

pub fn link<'tcx>(
    sess: &'tcx Session,
    codegen_results: &CodegenResults,
    outputs: &OutputFilenames,
    crate_name: &str,
) {
    debug!("Linking crate `{}`", crate_name);
    // largely inspired by rust-gpu
    let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
    for &crate_type in sess.crate_types().iter() {
        if (sess.opts.unstable_opts.no_codegen || !sess.opts.output_types.should_codegen())
            && !output_metadata
            && crate_type == CrateType::Executable
        {
            continue;
        }

        for obj in codegen_results
            .modules
            .iter()
            .filter_map(|m| m.object.as_ref())
        {
            check_file_is_writeable(obj, sess);
        }

        if outputs.outputs.should_codegen() {
            let out_filename = out_filename(
                sess,
                crate_type,
                outputs,
                codegen_results.crate_info.local_crate_name,
            );
            match crate_type {
                CrateType::Rlib => {
                    link_rlib(sess, codegen_results, &out_filename);
                }
                CrateType::Executable | CrateType::Cdylib | CrateType::Dylib => {
                    let _ = link_exe(
                        &codegen_results.allocator_module,
                        sess,
                        crate_type,
                        &out_filename,
                        codegen_results,
                    );
                }
                other => sess.fatal(&format!("Invalid crate type: {:?}", other)),
            }
        }
    }
}

fn link_rlib(sess: &Session, codegen_results: &CodegenResults, out_filename: &Path) {
    debug!("Linking rlib `{:?}`", out_filename);
    let mut file_list = Vec::<&Path>::new();

    for obj in codegen_results
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        file_list.push(obj);
    }

    for lib in codegen_results.crate_info.used_libraries.iter() {
        match lib.kind {
            NativeLibKind::Static {
                bundle: None | Some(true),
                ..
            } => {}
            NativeLibKind::Static {
                bundle: Some(false),
                ..
            }
            | NativeLibKind::Dylib { .. }
            | NativeLibKind::Framework { .. }
            | NativeLibKind::RawDylib
            | NativeLibKind::LinkArg
            | NativeLibKind::Unspecified => continue,
        }
        // native libraries in cuda doesnt make much sense, extern functions
        // do exist in nvvm for stuff like cuda syscalls and cuda provided functions
        // but including libraries doesnt make sense because nvvm would have to translate
        // the binary directly to ptx. We might want to add some way of linking in
        // ptx files or custom bitcode modules as "libraries" perhaps in the future.
        if let Some(name) = lib.name {
            sess.err(&format!(
                "Adding native libraries to rlib is not supported in CUDA: {}",
                name
            ));
        }
    }
    trace!("Files linked in rlib:\n{:#?}", file_list);

    create_archive(
        sess,
        &file_list,
        codegen_results.metadata.raw_data(),
        out_filename,
    );
}

fn link_exe(
    allocator: &Option<CompiledModule>,
    sess: &Session,
    crate_type: CrateType,
    out_filename: &Path,
    codegen_results: &CodegenResults,
) -> io::Result<()> {
    let mut objects = Vec::new();
    let mut rlibs = Vec::new();
    for obj in codegen_results
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        objects.push(obj.clone());
    }

    link_local_crate_native_libs_and_dependent_crate_libs(
        &mut rlibs,
        sess,
        crate_type,
        codegen_results,
    );

    let mut root_file_name = out_filename.file_name().unwrap().to_owned();
    root_file_name.push(".dir");
    let out_dir = out_filename.with_file_name(root_file_name);
    if !out_dir.is_dir() {
        std::fs::create_dir_all(&out_dir)?;
    }

    codegen_into_ptx_file(allocator, sess, &objects, &rlibs, out_filename)
}

/// This is the meat of the codegen, taking all of the llvm bitcode modules we have, and giving them to
/// nvvm to make into a final
fn codegen_into_ptx_file(
    allocator: &Option<CompiledModule>,
    sess: &Session,
    objects: &[PathBuf],
    rlibs: &[PathBuf],
    out_filename: &Path,
) -> io::Result<()> {
    debug!("Codegenning crate into PTX, allocator: {}, objects:\n{:#?}, rlibs:\n{:#?}, out_filename:\n{:#?}",
        allocator.is_some(),
        objects,
        rlibs,
        out_filename
    );

    // we need to make a new llvm context because we need it for linking together modules,
    // but we dont have our original one because rustc drops tyctxt and codegencx before linking.
    let cx = LlvmMod::new("link_tmp");

    let mut modules = Vec::with_capacity(objects.len() + rlibs.len());

    // object files (theyre not object files, they are impostors ඞ) are the bitcode modules produced by this codegen session
    // they *should* be the final crate.
    for obj in objects {
        let bitcode = std::fs::read(obj)?;
        modules.push(bitcode);
    }

    // rlibs are archives that we made previously, they are usually made for crates that are referenced
    // in this crate. We must unpack them and devour their bitcode to link in.
    for rlib in rlibs {
        let mut cgus = Vec::with_capacity(16);
        for entry in Archive::new(File::open(rlib)?).entries()? {
            let mut entry = entry?;
            // metadata is where rustc puts rlib metadata, so its not a cgu we are interested in.
            if entry.path().unwrap() != Path::new(".metadata") {
                // std::fs::read adds 1 to the size, so do the same here - see comment:
                // https://github.com/rust-lang/rust/blob/72868e017bdade60603a25889e253f556305f996/library/std/src/fs.rs#L200-L202
                let mut bitcode = Vec::with_capacity(entry.size() as usize + 1);
                entry.read_to_end(&mut bitcode).unwrap();
                cgus.push(bitcode);
            }
        }

        modules.extend(cgus);
    }

    if let Some(alloc) = allocator {
        let bc = std::fs::read(
            alloc
                .object
                .clone()
                .expect("expected obj path for allocator module"),
        )?;
        modules.push(bc);
    }

    // now that we have our nice bitcode modules, we just need to find libdevice and give our
    // modules to nvvm to make a final ptx file

    // we need to actually parse the codegen args again, because codegencx is not available at link time.
    let args = CodegenArgs::from_session(sess);

    let ptx_bytes = match crate::nvvm::codegen_bitcode_modules(&args, sess, modules, cx.llcx) {
        Ok(bytes) => bytes,
        Err(err) => {
            // TODO(RDambrosio016): maybe include the nvvm log with this fatal error
            sess.fatal(&err.to_string())
        }
    };

    std::fs::write(out_filename, ptx_bytes)
}

fn create_archive(sess: &Session, files: &[&Path], metadata: &[u8], out_filename: &Path) {
    if let Err(err) = try_create_archive(files, metadata, out_filename) {
        sess.fatal(&format!("Failed to create archive: {}", err));
    }
}

fn try_create_archive(files: &[&Path], metadata: &[u8], out_filename: &Path) -> io::Result<()> {
    let file = File::create(out_filename)?;
    let mut builder = Builder::new(file);
    {
        let mut header = Header::new_gnu();
        header.set_path(".metadata")?;
        header.set_size(metadata.len() as u64);
        header.set_cksum();
        builder.append(&header, metadata)?;
    }
    let mut filenames = FxHashSet::default();
    filenames.insert(OsStr::new(".metadata"));
    for file in files {
        assert!(
            filenames.insert(file.file_name().unwrap()),
            "Duplicate filename in archive: {:?}",
            file.file_name().unwrap()
        );
        builder.append_path_with_name(file, file.file_name().unwrap())?;
    }
    builder.into_inner()?;
    Ok(())
}

// most of the code from here is derived from rust-gpu

fn link_local_crate_native_libs_and_dependent_crate_libs<'a>(
    rlibs: &mut Vec<PathBuf>,
    sess: &'a Session,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
) {
    if sess.opts.unstable_opts.link_native_libraries {
        add_local_native_libraries(sess, codegen_results);
    }
    add_upstream_rust_crates(sess, rlibs, codegen_results, crate_type);
    if sess.opts.unstable_opts.link_native_libraries {
        add_upstream_native_libraries(sess, codegen_results, crate_type);
    }
}

fn add_local_native_libraries(sess: &Session, codegen_results: &CodegenResults) {
    let relevant_libs = codegen_results
        .crate_info
        .used_libraries
        .iter()
        .filter(|l| relevant_lib(sess, l));
    assert_eq!(relevant_libs.count(), 0);
}

fn add_upstream_rust_crates(
    sess: &Session,
    rlibs: &mut Vec<PathBuf>,
    codegen_results: &CodegenResults,
    crate_type: CrateType,
) {
    let (_, data) = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| *ty == crate_type)
        .expect("failed to find crate type in dependency format list");
    let deps = &codegen_results.crate_info.used_crates;
    for cnum in deps.iter() {
        let src = &codegen_results.crate_info.used_crate_source[cnum];
        match data[cnum.as_usize() - 1] {
            Linkage::NotLinked => {}
            Linkage::Static => rlibs.push(src.rlib.as_ref().unwrap().0.clone()),
            // should we just ignore includedFromDylib?
            Linkage::Dynamic | Linkage::IncludedFromDylib => {
                sess.fatal("Dynamic Linking is not supported in CUDA")
            }
        }
    }
}

fn add_upstream_native_libraries(
    sess: &Session,
    codegen_results: &CodegenResults,
    _crate_type: CrateType,
) {
    let crates = &codegen_results.crate_info.used_crates;
    for cnum in crates {
        for lib in codegen_results.crate_info.native_libraries[cnum].iter() {
            if !relevant_lib(sess, lib) {
                continue;
            }
            sess.fatal("Native libraries are not supported in CUDA");
        }
    }
}

fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => rustc_attr::cfg_matches(cfg, &sess.parse_sess, CRATE_NODE_ID, None),
        None => true,
    }
}
