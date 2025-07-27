//! High level safe bindings to the NVVM compiler (libnvvm) for writing CUDA GPU kernels with a subset of LLVM IR.

use std::{
    ffi::{CStr, CString},
    fmt::Display,
    mem::MaybeUninit,
    ptr::null_mut,
    str::FromStr,
};

use strum::IntoEnumIterator;

use cust_raw::nvvm_sys;

pub use cust_raw::nvvm_sys::LIBDEVICE_BITCODE;

/// Get the major and minor NVVM IR version.
pub fn ir_version() -> (i32, i32) {
    unsafe {
        let mut major_ir = MaybeUninit::uninit();
        let mut minor_ir = MaybeUninit::uninit();
        let mut major_dbg = MaybeUninit::uninit();
        let mut minor_dbg = MaybeUninit::uninit();
        // according to the docs this cant fail
        let _ = nvvm_sys::nvvmIRVersion(
            major_ir.as_mut_ptr(),
            minor_ir.as_mut_ptr(),
            major_dbg.as_mut_ptr(),
            minor_dbg.as_mut_ptr(),
        );
        (major_ir.assume_init(), minor_ir.assume_init())
    }
}

/// Get the major and minor NVVM debug metadata version.
pub fn dbg_version() -> (i32, i32) {
    unsafe {
        let mut major_ir = MaybeUninit::uninit();
        let mut minor_ir = MaybeUninit::uninit();
        let mut major_dbg = MaybeUninit::uninit();
        let mut minor_dbg = MaybeUninit::uninit();
        // according to the docs this cant fail
        let _ = nvvm_sys::nvvmIRVersion(
            major_ir.as_mut_ptr(),
            minor_ir.as_mut_ptr(),
            major_dbg.as_mut_ptr(),
            minor_dbg.as_mut_ptr(),
        );
        (major_dbg.assume_init(), minor_dbg.assume_init())
    }
}

/// Get the major and minor NVVM version.
pub fn nvvm_version() -> (i32, i32) {
    unsafe {
        let mut major = MaybeUninit::uninit();
        let mut minor = MaybeUninit::uninit();
        // according to the docs this cant fail
        let _ = nvvm_sys::nvvmVersion(major.as_mut_ptr(), minor.as_mut_ptr());
        (major.assume_init(), minor.assume_init())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvvmError {
    /// The NVVM compiler ran out of memory.
    OutOfMemory,
    /// The program could not be created for an unspecified reason.
    ProgramCreationFailure,
    IrVersionMismatch,
    InvalidInput,
    /// The IR given to the program was invalid. Getting the compiler
    /// log should yield more info.
    InvalidIr,
    /// A compile option given to the compiler was invalid.
    InvalidOption,
    /// The program has no modules OR all modules are lazy modules.
    NoModuleInProgram,
    /// Compilation failed because of bad IR or other reasons. Getting the compiler
    /// log should yield more info.
    CompilationError,
    // InvalidProgram isnt handled because its not possible
    // to get an invalid program handle through this safe api
}

impl Display for NvvmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = nvvm_sys::nvvmGetErrorString(self.to_raw());
            f.write_str(&CStr::from_ptr(ptr).to_string_lossy())
        }
    }
}

impl NvvmError {
    fn to_raw(self) -> nvvm_sys::nvvmResult {
        match self {
            NvvmError::CompilationError => nvvm_sys::nvvmResult::NVVM_ERROR_COMPILATION,
            NvvmError::OutOfMemory => nvvm_sys::nvvmResult::NVVM_ERROR_OUT_OF_MEMORY,
            NvvmError::ProgramCreationFailure => {
                nvvm_sys::nvvmResult::NVVM_ERROR_PROGRAM_CREATION_FAILURE
            }
            NvvmError::IrVersionMismatch => nvvm_sys::nvvmResult::NVVM_ERROR_IR_VERSION_MISMATCH,
            NvvmError::InvalidOption => nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_OPTION,
            NvvmError::InvalidInput => nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_INPUT,
            NvvmError::InvalidIr => nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_IR,
            NvvmError::NoModuleInProgram => nvvm_sys::nvvmResult::NVVM_ERROR_NO_MODULE_IN_PROGRAM,
        }
    }

    fn from_raw(result: nvvm_sys::nvvmResult) -> Self {
        use NvvmError::*;
        match result {
            nvvm_sys::nvvmResult::NVVM_ERROR_COMPILATION => CompilationError,
            nvvm_sys::nvvmResult::NVVM_ERROR_OUT_OF_MEMORY => OutOfMemory,
            nvvm_sys::nvvmResult::NVVM_ERROR_PROGRAM_CREATION_FAILURE => ProgramCreationFailure,
            nvvm_sys::nvvmResult::NVVM_ERROR_IR_VERSION_MISMATCH => IrVersionMismatch,
            nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_OPTION => InvalidOption,
            nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_INPUT => InvalidInput,
            nvvm_sys::nvvmResult::NVVM_ERROR_INVALID_IR => InvalidIr,
            nvvm_sys::nvvmResult::NVVM_ERROR_NO_MODULE_IN_PROGRAM => NoModuleInProgram,
            nvvm_sys::nvvmResult::NVVM_SUCCESS => panic!(),
            _ => unreachable!(),
        }
    }
}

trait ToNvvmResult {
    fn to_result(self) -> Result<(), NvvmError>;
}

impl ToNvvmResult for nvvm_sys::nvvmResult {
    fn to_result(self) -> Result<(), NvvmError> {
        let err = match self {
            nvvm_sys::nvvmResult::NVVM_SUCCESS => return Ok(()),
            _ => NvvmError::from_raw(self),
        };
        Err(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvvmOption {
    /// Generate debug info, valid only with an opt-level of `0` (`-g`).
    GenDebugInfo,
    /// Generate line number info (`-generate-line-info`).
    GenLineInfo,
    /// Whether to disable optimizations (opt level 0).
    NoOpts,
    /// The NVVM arch to use.
    Arch(NvvmArch),
    /// Whether to flush denormal values to zero when performing single-precision
    /// floating point operations. False by default.
    Ftz,
    /// Whether to use a fast approximation for sqrt instead of
    /// IEEE round-to-nearest mode for single-precision float square root.
    FastSqrt,
    /// Whether to use a fast approximation for div and reciprocal instead of
    /// IEEE round-to-nearest mode for single-precision float division.
    FastDiv,
    /// Whether to enable FMA contraction.
    NoFmaContraction,
}

impl Display for NvvmOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = match self {
            Self::GenDebugInfo => "-g",
            Self::GenLineInfo => "-generate-line-info",
            Self::NoOpts => "-opt=0",
            Self::Arch(arch) => return f.write_str(&format!("-arch={arch}")),
            Self::Ftz => "-ftz=1",
            Self::FastSqrt => "-prec-sqrt=0",
            Self::FastDiv => "-prec-div=0",
            Self::NoFmaContraction => "-fma=0",
        };
        f.write_str(res)
    }
}

impl FromStr for NvvmOption {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        Ok(match s {
            "-g" => Self::GenDebugInfo,
            "-generate-line-info" => Self::GenLineInfo,
            _ if s.starts_with("-opt=") => {
                let slice = &s[5..];
                if slice == "0" {
                    Self::NoOpts
                } else if slice == "3" {
                    // implied
                    return Err("-opt=3 is default");
                } else {
                    return Err("unknown optimization level");
                }
            }
            _ if s.starts_with("-ftz=") => {
                let slice = &s[5..];
                if slice == "1" {
                    Self::Ftz
                } else if slice == "0" {
                    // implied
                    return Err("-ftz=0 is default");
                } else {
                    return Err("unknown ftz option");
                }
            }
            _ if s.starts_with("-prec-sqrt=") => {
                let slice = &s[11..];
                if slice == "0" {
                    Self::FastSqrt
                } else if slice == "1" {
                    // implied
                    return Err("-prec-sqrt=1 is default");
                } else {
                    return Err("unknown prec-sqrt option");
                }
            }
            _ if s.starts_with("-prec-div=") => {
                let slice = &s[10..];
                if slice == "0" {
                    Self::FastDiv
                } else if slice == "1" {
                    // implied
                    return Err("-prec-div=1 is default");
                } else {
                    return Err("unknown prec-div option");
                }
            }
            _ if s.starts_with("-fma=") => {
                let slice = &s[5..];
                if slice == "0" {
                    Self::NoFmaContraction
                } else if slice == "1" {
                    // implied
                    return Err("-fma=1 is default");
                } else {
                    return Err("unknown fma option");
                }
            }
            _ if s.starts_with("-arch=") => {
                let slice = &s[6..];
                let arch_num = &slice[8..];
                let arch = match arch_num {
                    "35" => NvvmArch::Compute35,
                    "37" => NvvmArch::Compute37,
                    "50" => NvvmArch::Compute50,
                    "52" => NvvmArch::Compute52,
                    "53" => NvvmArch::Compute53,
                    "60" => NvvmArch::Compute60,
                    "61" => NvvmArch::Compute61,
                    "62" => NvvmArch::Compute62,
                    "70" => NvvmArch::Compute70,
                    "72" => NvvmArch::Compute72,
                    "75" => NvvmArch::Compute75,
                    "80" => NvvmArch::Compute80,
                    "86" => NvvmArch::Compute86,
                    "87" => NvvmArch::Compute87,
                    "89" => NvvmArch::Compute89,
                    "90" => NvvmArch::Compute90,
                    _ => return Err("unknown arch"),
                };
                Self::Arch(arch)
            }
            _ => return Err("umknown option"),
        })
    }
}

/// Nvvm architecture, default is `Compute52`
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
pub enum NvvmArch {
    Compute35,
    Compute37,
    Compute50,
    Compute52,
    Compute53,
    Compute60,
    Compute61,
    Compute62,
    Compute70,
    Compute72,
    Compute75,
    Compute80,
    Compute86,
    Compute87,
    Compute89,
    Compute90,
}

impl Display for NvvmArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut raw = format!("{self:?}").to_ascii_lowercase();
        raw.insert(7, '_');
        f.write_str(&raw)
    }
}

impl Default for NvvmArch {
    fn default() -> Self {
        Self::Compute52
    }
}

impl NvvmArch {
    /// Get the numeric capability value (e.g., 35 for Compute35)
    pub fn capability_value(&self) -> u32 {
        match self {
            Self::Compute35 => 35,
            Self::Compute37 => 37,
            Self::Compute50 => 50,
            Self::Compute52 => 52,
            Self::Compute53 => 53,
            Self::Compute60 => 60,
            Self::Compute61 => 61,
            Self::Compute62 => 62,
            Self::Compute70 => 70,
            Self::Compute72 => 72,
            Self::Compute75 => 75,
            Self::Compute80 => 80,
            Self::Compute86 => 86,
            Self::Compute87 => 87,
            Self::Compute89 => 89,
            Self::Compute90 => 90,
        }
    }

    /// Get the target feature string (e.g., "compute_35" for Compute35)
    pub fn target_feature(&self) -> String {
        let cap = self.capability_value();
        format!("compute_{cap}")
    }

    /// Get all target features up to and including this architecture.
    /// This ensures that `cfg(target_feature = "compute_50")` works on compute_60+ devices.
    pub fn all_target_features(&self) -> Vec<String> {
        let current = self.capability_value();

        NvvmArch::iter()
            .filter(|arch| arch.capability_value() <= current)
            .map(|arch| arch.target_feature())
            .collect()
    }

    /// Create an iterator over all architectures from Compute35 up to and including this one
    pub fn iter_up_to(&self) -> impl Iterator<Item = Self> {
        let current = self.capability_value();
        NvvmArch::iter().filter(move |arch| arch.capability_value() <= current)
    }
}

pub struct NvvmProgram {
    raw: nvvm_sys::nvvmProgram,
}

impl Drop for NvvmProgram {
    fn drop(&mut self) {
        unsafe {
            nvvm_sys::nvvmDestroyProgram(&mut self.raw as *mut _)
                .to_result()
                .expect("failed to destroy nvvm program");
        }
    }
}

impl NvvmProgram {
    /// Make a new NVVM program.
    pub fn new() -> Result<Self, NvvmError> {
        unsafe {
            let mut raw = MaybeUninit::uninit();
            nvvm_sys::nvvmCreateProgram(raw.as_mut_ptr()).to_result()?;
            Ok(Self {
                raw: raw.assume_init(),
            })
        }
    }

    /// Compile this program into PTX assembly bytes (they *should* be ascii per the PTX ISA ref but they are returned as bytes to be safe).
    ///
    pub fn compile(&self, options: &[NvvmOption]) -> Result<Vec<u8>, NvvmError> {
        unsafe {
            let options = options.iter().map(|x| format!("{x}\0")).collect::<Vec<_>>();
            let mut options_ptr = options
                .iter()
                .map(|x| x.as_ptr().cast())
                .collect::<Vec<_>>();

            nvvm_sys::nvvmCompileProgram(self.raw, options.len() as i32, options_ptr.as_mut_ptr())
                .to_result()?;
            let mut size = 0;
            nvvm_sys::nvvmGetCompiledResultSize(self.raw, &mut size as *mut usize as *mut _)
                .to_result()?;
            let mut buf: Vec<u8> = Vec::with_capacity(size);
            nvvm_sys::nvvmGetCompiledResult(self.raw, buf.as_mut_ptr().cast()).to_result()?;
            buf.set_len(size);
            // 𝖇𝖆𝖓𝖎𝖘𝖍 𝖙𝖍𝖞 𝖓𝖚𝖑
            buf.pop();
            Ok(buf)
        }
    }

    /// Add a bitcode module to this nvvm program.
    pub fn add_module(&self, bitcode: &[u8], name: String) -> Result<(), NvvmError> {
        unsafe {
            let cstring = CString::new(name).expect("module name with nul");
            nvvm_sys::nvvmAddModuleToProgram(
                self.raw,
                bitcode.as_ptr().cast(),
                bitcode.len(),
                cstring.as_ptr(),
            )
            .to_result()
        }
    }

    /// Add a bitcode module lazily to this nvvm program. This means that a symbol in this module
    /// is only loaded if it is used by a previous module. According to libnvvm docs, this also
    /// makes the symbols internal to the NVVM IR module, allowing for further optimizations.
    ///
    /// **Do not feed LLVM IR to this method, [`add_module`](Self::add_module) seems to allow it for now, but
    /// it yields an empty ptx file if given to this method**
    pub fn add_lazy_module(&self, bitcode: &[u8], name: String) -> Result<(), NvvmError> {
        unsafe {
            let cstring = CString::new(name).expect("module name with nul");
            nvvm_sys::nvvmLazyAddModuleToProgram(
                self.raw,
                bitcode.as_ptr().cast(),
                bitcode.len(),
                cstring.as_ptr(),
            )
            .to_result()
        }
    }

    /// Get the compiler/verifier log message. This includes any errors that may have happened during compilation
    /// or during verification as well as any warnings. If you are having trouble with your program yielding a
    /// compilation error, looking at this log *after* attempting compilation should help.
    ///
    /// Returns `None` if the log is empty and automatically strips off the nul at the end of the log.
    pub fn compiler_log(&self) -> Result<Option<String>, NvvmError> {
        unsafe {
            let mut size = MaybeUninit::uninit();
            nvvm_sys::nvvmGetProgramLogSize(self.raw, size.as_mut_ptr()).to_result()?;
            let size = size.assume_init();
            let mut buf: Vec<u8> = Vec::with_capacity(size);
            nvvm_sys::nvvmGetProgramLog(self.raw, buf.as_mut_ptr().cast()).to_result()?;
            buf.set_len(size);
            // 𝖇𝖆𝖓𝖎𝖘𝖍 𝖙𝖍𝖞 𝖓𝖚𝖑
            buf.pop();
            let string = String::from_utf8(buf).expect("nvvm compiler log was not utf8");
            Ok(Some(string).filter(|s| !s.is_empty()))
        }
    }

    /// Verify the program without actually compiling it. In the case of invalid IR, you can find
    /// more detailed error info by calling [`compiler_log`](Self::compiler_log).
    pub fn verify(&self) -> Result<(), NvvmError> {
        unsafe { nvvm_sys::nvvmVerifyProgram(self.raw, 0, null_mut()).to_result() }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    #[test]
    fn nvvm_arch_capability_value() {
        use crate::NvvmArch;

        assert_eq!(NvvmArch::Compute35.capability_value(), 35);
        assert_eq!(NvvmArch::Compute37.capability_value(), 37);
        assert_eq!(NvvmArch::Compute50.capability_value(), 50);
        assert_eq!(NvvmArch::Compute52.capability_value(), 52);
        assert_eq!(NvvmArch::Compute53.capability_value(), 53);
        assert_eq!(NvvmArch::Compute60.capability_value(), 60);
        assert_eq!(NvvmArch::Compute61.capability_value(), 61);
        assert_eq!(NvvmArch::Compute62.capability_value(), 62);
        assert_eq!(NvvmArch::Compute70.capability_value(), 70);
        assert_eq!(NvvmArch::Compute72.capability_value(), 72);
        assert_eq!(NvvmArch::Compute75.capability_value(), 75);
        assert_eq!(NvvmArch::Compute80.capability_value(), 80);
        assert_eq!(NvvmArch::Compute86.capability_value(), 86);
        assert_eq!(NvvmArch::Compute87.capability_value(), 87);
        assert_eq!(NvvmArch::Compute89.capability_value(), 89);
        assert_eq!(NvvmArch::Compute90.capability_value(), 90);
    }

    #[test]
    fn nvvm_arch_target_feature_format() {
        use crate::NvvmArch;

        assert_eq!(NvvmArch::Compute35.target_feature(), "compute_35");
        assert_eq!(NvvmArch::Compute61.target_feature(), "compute_61");
        assert_eq!(NvvmArch::Compute90.target_feature(), "compute_90");
    }

    #[test]
    fn nvvm_arch_all_target_features_includes_lower_capabilities() {
        use crate::NvvmArch;

        // Compute35 only includes itself
        let compute35_features = NvvmArch::Compute35.all_target_features();
        assert_eq!(compute35_features, vec!["compute_35"]);

        // Compute50 includes all lower capabilities
        let compute50_features = NvvmArch::Compute50.all_target_features();
        assert_eq!(
            compute50_features,
            vec!["compute_35", "compute_37", "compute_50"]
        );

        // Compute61 includes all lower capabilities
        let compute61_features = NvvmArch::Compute61.all_target_features();
        assert_eq!(
            compute61_features,
            vec![
                "compute_35",
                "compute_37",
                "compute_50",
                "compute_52",
                "compute_53",
                "compute_60",
                "compute_61"
            ]
        );

        // Compute90 includes all capabilities
        let compute90_features = NvvmArch::Compute90.all_target_features();
        assert_eq!(
            compute90_features,
            vec![
                "compute_35",
                "compute_37",
                "compute_50",
                "compute_52",
                "compute_53",
                "compute_60",
                "compute_61",
                "compute_62",
                "compute_70",
                "compute_72",
                "compute_75",
                "compute_80",
                "compute_86",
                "compute_87",
                "compute_89",
                "compute_90"
            ]
        );
    }

    #[test]
    fn target_feature_synthesis_supports_conditional_compilation_patterns() {
        use crate::NvvmArch;

        // When targeting Compute61, should enable all lower capabilities
        let features = NvvmArch::Compute61.all_target_features();

        // Should enable compute_60 (for f64 atomics)
        assert!(features.contains(&"compute_60".to_string()));

        // Should enable compute_50 (for 64-bit integer atomics)
        assert!(features.contains(&"compute_50".to_string()));

        // Should enable compute_35 (baseline)
        assert!(features.contains(&"compute_35".to_string()));

        // Should enable the target itself
        assert!(features.contains(&"compute_61".to_string()));

        // Should NOT enable higher capabilities
        assert!(!features.contains(&"compute_62".to_string()));
        assert!(!features.contains(&"compute_70".to_string()));
    }

    #[test]
    fn target_feature_synthesis_enables_correct_cfg_patterns() {
        use crate::NvvmArch;

        // Test that targeting Compute70 enables appropriate cfg patterns
        let features = NvvmArch::Compute70.all_target_features();

        // These should all be true for compute_70 target
        let expected_enabled = [
            "compute_35",
            "compute_37",
            "compute_50",
            "compute_52",
            "compute_53",
            "compute_60",
            "compute_61",
            "compute_62",
            "compute_70",
        ];

        for feature in expected_enabled {
            assert!(
                features.contains(&feature.to_string()),
                "Compute70 should enable {feature} for cfg(target_feature = \"{feature}\")"
            );
        }

        // These should NOT be enabled for compute_70 target
        let expected_disabled = ["compute_72", "compute_75", "compute_80", "compute_90"];

        for feature in expected_disabled {
            assert!(
                !features.contains(&feature.to_string()),
                "Compute70 should NOT enable {feature}"
            );
        }
    }

    #[test]
    fn nvvm_arch_iter_up_to_includes_only_lower_or_equal() {
        use crate::NvvmArch;

        // Compute35 only includes itself
        let archs: Vec<_> = NvvmArch::Compute35.iter_up_to().collect();
        assert_eq!(archs, vec![NvvmArch::Compute35]);

        // Compute52 includes all up to 52
        let archs: Vec<_> = NvvmArch::Compute52.iter_up_to().collect();
        assert_eq!(
            archs,
            vec![
                NvvmArch::Compute35,
                NvvmArch::Compute37,
                NvvmArch::Compute50,
                NvvmArch::Compute52,
            ]
        );

        // Compute75 includes all up to 75
        let archs: Vec<_> = NvvmArch::Compute75.iter_up_to().collect();
        assert_eq!(
            archs,
            vec![
                NvvmArch::Compute35,
                NvvmArch::Compute37,
                NvvmArch::Compute50,
                NvvmArch::Compute52,
                NvvmArch::Compute53,
                NvvmArch::Compute60,
                NvvmArch::Compute61,
                NvvmArch::Compute62,
                NvvmArch::Compute70,
                NvvmArch::Compute72,
                NvvmArch::Compute75,
            ]
        );
    }

    #[test]
    fn options_parse_correctly() {
        use crate::NvvmArch::*;
        use crate::NvvmOption::{self, *};

        let opts = vec![
            "-g",
            "-generate-line-info",
            "-opt=0",
            "-arch=compute_35",
            "-arch=compute_37",
            "-arch=compute_50",
            "-arch=compute_52",
            "-arch=compute_53",
            "-arch=compute_60",
            "-arch=compute_61",
            "-arch=compute_62",
            "-arch=compute_70",
            "-arch=compute_72",
            "-arch=compute_75",
            "-arch=compute_80",
            "-ftz=1",
            "-prec-sqrt=0",
            "-prec-div=0",
            "-fma=0",
        ];
        let expected = vec![
            GenDebugInfo,
            GenLineInfo,
            NoOpts,
            Arch(Compute35),
            Arch(Compute37),
            Arch(Compute50),
            Arch(Compute52),
            Arch(Compute53),
            Arch(Compute60),
            Arch(Compute61),
            Arch(Compute62),
            Arch(Compute70),
            Arch(Compute72),
            Arch(Compute75),
            Arch(Compute80),
            Ftz,
            FastSqrt,
            FastDiv,
            NoFmaContraction,
        ];

        let found = opts
            .into_iter()
            .map(|x| NvvmOption::from_str(x).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(found, expected);
    }
}
