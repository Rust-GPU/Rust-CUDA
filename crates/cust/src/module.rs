//! Functions and types for working with CUDA modules.

use crate::error::{CudaResult, DropResult, ToResult};
use crate::function::Function;
use crate::memory::{CopyDestination, DeviceCopy, DevicePointer};
use crate::sys as cuda;
use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::path::Path;
use std::ptr;

/// A compiled CUDA module, loaded into a context.
#[derive(Debug)]
pub struct Module {
    inner: cuda::CUmodule,
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

    /// Load a module from a normal (rust) string, implicitly making it into
    /// a cstring.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<T: AsRef<str>>(string: T) -> CudaResult<Module> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
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
    /// let name = CString::new("sum")?;
    /// let function = module.get_function(&name)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_function<'a, T: AsRef<str>>(&'a self, name: T) -> CudaResult<Function<'a>> {
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
                cuda::cuMemcpyHtoD_v2(
                    self.ptr.as_raw_mut() as u64,
                    val as *const T as *const c_void,
                    size,
                )
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
