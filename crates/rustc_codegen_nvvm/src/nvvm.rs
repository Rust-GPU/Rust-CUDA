//! Final steps in codegen, coalescing modules and feeding them to libnvvm.
//!
//! This module also includes a safe wrapper over the nvvm_sys module.

use nvvm::*;
use rustc_session::Session;
use std::env;
use std::ffi::OsStr;
use std::fmt::Display;
use std::fs;
use std::path::Path;
use tracing::debug;

// see libintrinsics.ll on what this is.
const LIBINTRINSICS: &[u8] = include_bytes!("../libintrinsics.bc");

pub enum CodegenErr {
    Nvvm(NvvmError),
    Io(std::io::Error),
}

impl From<std::io::Error> for CodegenErr {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}

impl From<NvvmError> for CodegenErr {
    fn from(v: NvvmError) -> Self {
        Self::Nvvm(v)
    }
}

impl Display for CodegenErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvvm(err) => std::fmt::Display::fmt(&err, f),
            Self::Io(err) => std::fmt::Display::fmt(&err, f),
        }
    }
}

/// Take a list of bitcode module bytes and their names and codegen it
/// into ptx bytes. The final PTX *should* be utf8, but just to be on the safe side
/// it returns a vector of bytes.
///
/// Note that this will implicitly try to find libdevice and add it, so don't do that
/// step before this. It will fatal error if it cannot find it.
pub fn codegen_bitcode_modules(
    opts: &[NvvmOption],
    sess: &Session,
    main: Vec<(Vec<u8>, String)>,
    lazy: Vec<(Vec<u8>, String)>,
) -> Result<Vec<u8>, CodegenErr> {
    debug!(
        "Codegenning bitcode, main modules:\n{:?}, lazy loaded modules:\n{:?}",
        main, lazy
    );
    // first, create the nvvm program we will add modules to.
    let prog = NvvmProgram::new()?;

    // next, load our main bitcode modules from the tempdir.
    for (bc, name) in main {
        prog.add_module(&bc, name)?;
    }

    // then, load our lazy bitcode modules.
    for (bc, name) in lazy {
        prog.add_lazy_module(&bc, name)?;
    }

    let libdevice = if let Some(bc) = find_libdevice() {
        bc
    } else {
        // i would put a more helpful error here, but to actually use the codegen
        // it needs to find libnvvm before this, and libdevice is in the nvvm directory
        // so if it can find libnvvm there is almost no way it can't find libdevice.
        sess.fatal("Could not find the libdevice library (libdevice.10.bc) in the CUDA directory")
    };

    prog.add_lazy_module(&libdevice, "libdevice".to_string())?;
    prog.add_lazy_module(&LIBINTRINSICS, "libintrinsics".to_string())?;

    // for now, while the codegen is young, we always run verification on the program.
    // This is to make debugging much easier, libnvvm tends to infinitely loop or segfault on invalid programs
    // which makes debugging extremely hard. This way, if a malformed program is created, it is caught before
    // giving it to libnvvm. Then to debug codegen failures, we can just ask the user to provide the corresponding llvm ir
    // file with --emit=llvm-ir

    let verification_res = prog.verify();
    if verification_res.is_err() {
        let log = prog.compiler_log().unwrap().unwrap_or_default();
        let footer = "If you plan to submit a bug report please re-run the codegen with `RUSTFLAGS=\"--emit=llvm-ir\" and include the .ll file corresponding to the .o file mentioned in the log";
        panic!(
            "Malformed NVVM IR program rejected by libnvvm, dumping verifier log:\n\n{}\n\n{}",
            log, footer
        );
    }

    let res = match prog.compile(opts) {
        Ok(b) => b,
        Err(_) => {
            // this should never happen, if it does, something went really bad or its a bug on libnvvm's end
            panic!("libnvvm returned an error that was not previously caught by the verifier");
        }
    };

    Ok(res)
}

/// Find the libdevice bitcode library which contains math intrinsics and is linked
/// when building the nvvm program.
pub fn find_libdevice() -> Option<Vec<u8>> {
    #[cfg(windows)]
    {
        let base_path = Path::new(&env::var("CUDA_PATH").ok()?).to_path_buf();

        // we want libdevice.10.bc, however, i am not quite sure if the libdevice number can change,
        // so for safety, search the dir for a .bc file.
        let libdevice_file = fs::read_dir(base_path.join("nvvm").join("libdevice"))
            .ok()?
            .filter_map(Result::ok)
            .filter(|f| f.path().extension() == Some(OsStr::new("bc")))
            .next()?
            .path();

        fs::read(libdevice_file).ok()
    }
    #[cfg(not(windows))]
    {
        todo!();
        // // on some distros it seems to be in /usr/local/
        // let mut candidates = read_env();
        // candidates.push(PathBuf::from("/opt/cuda"));
        // candidates.push(PathBuf::from("/usr/local/cuda"));
        // for e in glob("/usr/local/cuda-*").unwrap() {
        //     if let Ok(path) = e {
        //         candidates.push(path)
        //     }
        // }
    }
}
