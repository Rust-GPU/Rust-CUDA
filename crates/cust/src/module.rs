//! Functions and types for working with CUDA modules.

use crate::error::{CudaResult, DropResult, ToResult};
use crate::function::Function;
use crate::memory::{CopyDestination, DeviceCopy, DevicePointer};
use crate::sys as cuda;
use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use std::path::Path;
use std::ptr;

/// A compiled CUDA module, loaded into a context.
#[derive(Debug)]
pub struct Module {
    inner: cuda::CUmodule,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// The possible optimization levels when JIT compiling a PTX module. `O4` by default (most optimized).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
    O4 = 4,
}

/// The possible targets when JIT compiling a PTX module.
#[non_exhaustive]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitTarget {
    Compute20 = 20,
    Compute21 = 21,
    Compute30 = 30,
    Compute32 = 32,
    Compute35 = 35,
    Compute37 = 37,
    Compute50 = 50,
    Compute52 = 52,
    Compute53 = 53,
    Compute60 = 60,
    Compute61 = 61,
    Compute62 = 62,
    Compute70 = 70,
    Compute72 = 72,
    Compute75 = 75,
    Compute80 = 80,
    Compute86 = 86,
}

/// How to handle cases where a loaded module's data does not contain an exact match for the
/// specified architecture.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitFallback {
    /// Prefer to compile PTX if present if an exact binary match is not found.
    PreferPtx = 0,
    /// Prefer to fall back to a compatible binary code match if exact match is not found.
    /// This means the driver may pick binary code for `7.0` if your device is `7.2` for example.
    PreferCompatibleBinary = 1,
}

/// Different options that could be applied when loading a module.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModuleJitOption {
    /// Specifies the maximum amount of registers any compiled PTX is allowed to use.
    MaxRegisters(u32),
    /// Specifies the optimization level for the JIT compiler.
    OptLevel(OptLevel),
    /// Determines the PTX target from the current context's architecture. Cannot be combined with
    /// [`ModuleJitOption::Target`].
    DetermineTargetFromContext,
    /// Specifies the target for the JIT compiler. Cannot be combined with [`ModuleJitOption::DetermineTargetFromContext`].
    Target(JitTarget),
    /// Specifies how to handle cases where a loaded module's data does not have an exact match for the specified
    /// architecture.
    Fallback(JitFallback),
    /// Generates debug info in the compiled binary.
    GenenerateDebugInfo(bool),
    /// Generates line info in the compiled binary.
    GenerateLineInfo(bool),
}

impl ModuleJitOption {
    pub fn into_raw(opts: &[Self]) -> (Vec<cuda::CUjit_option>, Vec<*mut c_void>) {
        let mut raw_opts = Vec::with_capacity(opts.len());
        let mut raw_vals = Vec::with_capacity(opts.len());
        for opt in opts {
            match opt {
                Self::MaxRegisters(regs) => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_MAX_REGISTERS);
                    raw_vals.push(regs as *const u32 as *mut _);
                }
                Self::OptLevel(level) => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_OPTIMIZATION_LEVEL);
                    raw_vals.push(level as *const OptLevel as *mut _);
                }
                Self::DetermineTargetFromContext => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_TARGET_FROM_CUCONTEXT);
                }
                Self::Target(target) => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_TARGET);
                    raw_vals.push(target as *const JitTarget as *mut _);
                }
                Self::Fallback(fallback) => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_FALLBACK_STRATEGY);
                    raw_vals.push(fallback as *const JitFallback as *mut _);
                }
                Self::GenenerateDebugInfo(gen) => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_GENERATE_DEBUG_INFO);
                    raw_vals.push(gen as *const bool as *mut _);
                }
                Self::GenerateLineInfo(gen) => {
                    raw_opts.push(cuda::CUjit_option::CU_JIT_GENERATE_LINE_INFO);
                    raw_vals.push(gen as *const bool as *mut _)
                }
            }
        }
        (raw_opts, raw_vals)
    }
}

#[cfg(unix)]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    path.as_ref().to_string_lossy().to_string().into_bytes()
}

impl Module {
    /// Load a module from the given path into the current context.
    ///
    /// The given path should be either a cubin file, a ptx file, or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let module = Module::from_file("./resources/add.ptx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> CudaResult<Module> {
        unsafe {
            let mut bytes = path_to_bytes(path);
            if !bytes.contains(&0) {
                bytes.push(0);
            }
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            cuda::cuModuleLoad(
                &mut module.inner as *mut cuda::CUmodule,
                bytes.as_ptr() as *const _,
            )
            .to_result()?;
            Ok(module)
        }
    }

    // TODO(RDambrosio016): figure out why the heck cuda rejects cubins literally made by nvcc and loaded by fs::read

    // /// Creates a new module by loading a fatbin (fat binary) file.
    // ///
    // /// Fatbinary files are files that contain multiple ptx or cubin files. The driver will choose already-built
    // /// cubin if it is present, and otherwise JIT compile any PTX in the file to cubin.
    // ///
    // /// # Example
    // ///
    // /// ```
    // /// # use cust::*;
    // /// # use std::error::Error;
    // /// # fn main() -> Result<(), Box<dyn Error>> {
    // /// # let _ctx = quick_init()?;
    // /// use cust::module::Module;
    // /// let fatbin_bytes = std::fs::read("./resources/add.cubin")?;
    // /// assert!(fatbin_bytes.contains(&0));
    // /// let module = Module::from_cubin(&fatbin_bytes, &[])?;
    // /// # Ok(())
    // /// # }
    // /// ```
    // pub fn from_fatbin<T: AsRef<[u8]>>(
    //     bytes: T,
    //     options: &[ModuleJitOption],
    // ) -> CudaResult<Module> {
    //     let mut bytes = bytes.as_ref().to_vec();
    //     bytes.push(0);
    //     // fatbins are just ELF files like cubins, and cuModuleLoadDataEx accepts ptx, cubin, and fatbin.
    //     // We just make the distinction in case we want to do anything extra in the future. As well
    //     // as keep things explicit to anyone reading the code.
    //     Self::from_cubin(bytes, options)
    // }

    // pub unsafe fn from_fatbin_unchecked<T: AsRef<[u8]>>(
    //     bytes: T,
    //     options: &[ModuleJitOption],
    // ) -> CudaResult<Module> {
    //     Self::from_cubin_unchecked(bytes, options)
    // }

    // pub fn from_cubin<T: AsRef<[u8]>>(bytes: T, options: &[ModuleJitOption]) -> CudaResult<Module> {
    //     let bytes = bytes.as_ref();
    //     goblin::elf::Elf::parse(bytes).expect("Cubin/Fatbin was not valid ELF!");
    //     // SAFETY: we verified the bytes were valid ELF
    //     unsafe { Self::from_cubin_unchecked(bytes, options) }
    // }

    // pub unsafe fn from_cubin_unchecked<T: AsRef<[u8]>>(
    //     bytes: T,
    //     options: &[ModuleJitOption],
    // ) -> CudaResult<Module> {
    //     let bytes = bytes.as_ref();
    //     let mut module = Module {
    //         inner: ptr::null_mut(),
    //     };
    //     let (mut options, mut option_values) = ModuleJitOption::into_raw(options);
    //     cuda::cuModuleLoadDataEx(
    //         &mut module.inner as *mut cuda::CUmodule,
    //         bytes.as_ptr() as *const c_void,
    //         options.len() as c_uint,
    //         options.as_mut_ptr(),
    //         option_values.as_mut_ptr(),
    //     )
    //     .to_result()?;
    //     Ok(module)
    // }

    pub fn from_ptx_cstr(cstr: &CStr, options: &[ModuleJitOption]) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            let (mut options, mut option_values) = ModuleJitOption::into_raw(options);
            cuda::cuModuleLoadDataEx(
                &mut module.inner as *mut cuda::CUmodule,
                cstr.as_ptr() as *const c_void,
                options.len() as c_uint,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            )
            .to_result()?;
            Ok(module)
        }
    }

    pub fn from_ptx<T: AsRef<str>>(string: T, options: &[ModuleJitOption]) -> CudaResult<Module> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
        Self::from_ptx_cstr(cstr.as_c_str(), options)
    }

    /// Load a module from a normal (rust) string, implicitly making it into
    /// a cstring.
    #[deprecated(
        since = "0.3.0",
        note = "from_str was too generic of a name, use from_ptx instead, passing an empty slice of options (usually)"
    )]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<T: AsRef<str>>(string: T) -> CudaResult<Module> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
        #[allow(deprecated)]
        Self::load_from_string(cstr.as_c_str())
    }

    /// Load a module from a CStr.
    ///
    /// This is useful in combination with `include_str!`, to include the device code into the
    /// compiled executable.
    ///
    /// The given CStr must contain the bytes of a cubin file, a ptx file or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let image = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&image)?;
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "0.3.0",
        note = "load_from_string was an inconsistent name with inconsistent params, use from_ptx/from_ptx_cstr, passing 
    an empty slice of options (usually)
    "
    )]
    pub fn load_from_string(image: &CStr) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            cuda::cuModuleLoadData(
                &mut module.inner as *mut cuda::CUmodule,
                image.as_ptr() as *const c_void,
            )
            .to_result()?;
            Ok(module)
        }
    }

    /// Get a reference to a global symbol, which can then be copied to/from.
    ///
    /// # Panics:
    ///
    /// This function panics if the size of the symbol is not the same as the `mem::sizeof<T>()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use cust::memory::CopyDestination;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// let name = CString::new("my_constant")?;
    /// let symbol = module.get_global::<u32>(&name)?;
    /// let mut host_const = 0;
    /// symbol.copy_to(&mut host_const)?;
    /// assert_eq!(314, host_const);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_global<'a, T: DeviceCopy>(&'a self, name: &CStr) -> CudaResult<Symbol<'a, T>> {
        unsafe {
            let mut ptr: DevicePointer<T> = DevicePointer::null();
            let mut size: usize = 0;

            cuda::cuModuleGetGlobal_v2(
                &mut ptr as *mut DevicePointer<T> as *mut cuda::CUdeviceptr,
                &mut size as *mut usize,
                self.inner,
                name.as_ptr(),
            )
            .to_result()?;
            assert_eq!(size, mem::size_of::<T>());
            Ok(Symbol {
                ptr,
                module: PhantomData,
            })
        }
    }

    /// Get a reference to a kernel function which can then be launched.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// let function = module.get_function("sum")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_function<T: AsRef<str>>(&'_ self, name: T) -> CudaResult<Function<'_>> {
        unsafe {
            let name = name.as_ref();
            let cstr = CString::new(name).expect("Argument to get_function had a nul");
            let mut func: cuda::CUfunction = ptr::null_mut();

            cuda::cuModuleGetFunction(
                &mut func as *mut cuda::CUfunction,
                self.inner,
                cstr.as_ptr(),
            )
            .to_result()?;
            Ok(Function::new(func, self))
        }
    }

    /// Destroy a `Module`, returning an error.
    ///
    /// Destroying a module can return errors from previous asynchronous work. This function
    /// destroys the given module and returns the error and the un-destroyed module on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use cust::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// match Module::drop(module) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, module)) => {
    ///         println!("Failed to destroy module: {:?}", e);
    ///         // Do something with module
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop(mut module: Module) -> DropResult<Module> {
        if module.inner.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut module.inner, ptr::null_mut());
            match cuda::cuModuleUnload(inner).to_result() {
                Ok(()) => {
                    mem::forget(module);
                    Ok(())
                }
                Err(e) => Err((e, Module { inner })),
            }
        }
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }
        unsafe {
            // No choice but to panic if this fails...
            let module = mem::replace(&mut self.inner, ptr::null_mut());
            cuda::cuModuleUnload(module);
        }
    }
}

/// Handle to a symbol defined within a CUDA module.
#[derive(Debug)]
pub struct Symbol<'a, T: DeviceCopy> {
    ptr: DevicePointer<T>,
    module: PhantomData<&'a Module>,
}
impl<'a, T: DeviceCopy> crate::private::Sealed for Symbol<'a, T> {}
impl<'a, T: DeviceCopy> fmt::Pointer for Symbol<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<'a, T: DeviceCopy> CopyDestination<T> for Symbol<'a, T> {
    fn copy_from(&mut self, val: &T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyHtoD_v2(self.ptr.as_raw(), val as *const T as *const c_void, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoH_v2(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as u64,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}
