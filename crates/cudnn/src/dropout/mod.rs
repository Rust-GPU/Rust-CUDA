mod dropout_descriptor;

pub use dropout_descriptor::*;

use crate::{sys, CudnnContext, CudnnError, DataType, IntoResult, TensorDescriptor};
use cust::memory::GpuBuffer;
use std::mem::MaybeUninit;

impl CudnnContext {
    /// This function is used to query the amount of space required to store the states of the
    /// random number generators.
    pub fn get_dropout_states_size(&self) -> Result<usize, CudnnError> {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnDropoutGetStatesSize(self.raw, size.as_mut_ptr()).into_result()?;

            Ok(size.assume_init())
        }
    }

    /// This function is used to query the amount of reserve needed to run dropout with the input
    /// dimensions given by `x_desc`.
    ///
    /// The same reserve space is expected to be passed to
    /// [`dropout_forward()`](CudnnContext::dropoit_forward()) and
    /// [`dropout_backward()`](CudnnContext::dropout_backward()).
    ///
    /// **Do note** that the content of there reserved space is expected to remain unchanged
    /// between both calls.
    ///
    /// # Arguments
    ///
    /// `x_desc` - a previously initialized tensor descriptor, describing input to a dropout
    /// operation.
    pub fn get_dropout_reserved_space_size<T>(
        &self,
        x_desc: &TensorDescriptor<T>,
    ) -> Result<usize, CudnnError>
    where
        T: DataType,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnDropoutGetStatesSize(self.raw, size.as_mut_ptr()).into_result()?;

            Ok(size.assume_init())
        }
    }

    /// Creates and initializes a generic dropout descriptor.
    ///
    /// # Arguments
    ///
    /// * `dropout` - probability with which the value from input is set to zero during the dropout
    /// layer.
    ///
    /// * `states` - user-allocated GPU memory that will hold random number generator states.
    ///
    /// * `seed` - seed used to initialize random number generator states.
    ///
    /// **Do note** that the exact amount of memory can be obtained with
    /// [`get_dropout_states_size()`](CudnnContext::get_dropout_states_sizes).
    ///
    /// # Errors
    ///
    /// Return errors if `states` size is less than that returned by `get_dropout_states_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let size = ctx.get_dropout_states_size()?;
    /// let states = unsafe { DeviceBuffer::uninitialized(size)? };
    ///
    /// let dropout = 0.5;
    /// let seed = 123;
    ///
    /// let dropout_desc = ctx.create_dropout_descriptor(dropout, states, seed)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_dropout_descriptor<T: GpuBuffer<u8>>(
        &self,
        dropout: f32,
        states: T,
        seed: u64,
    ) -> Result<DropoutDescriptor<T>, CudnnError> {
        let mut raw = MaybeUninit::uninit();
        let states_ptr = states.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let states_size = states.len();

        unsafe {
            sys::cudnnCreateDropoutDescriptor(raw.as_mut_ptr()).into_result()?;

            let mut raw = raw.assume_init();

            sys::cudnnSetDropoutDescriptor(raw, self.raw, dropout, states_ptr, states_size, seed)
                .into_result()?;

            Ok(DropoutDescriptor { raw, states })
        }
    }

    /// This function performs forward dropout operation over `x_data` returning results in
    /// `y_data`.
    ///
    /// The approximate dropout fraction of `x_data` values will be replaced by a 0, and the rest
    /// will be scaled by 1 / (1 - dropout), i.e. the value configured in `dropout_desc`.
    ///
    /// This function should not be running concurrently with another `dropout_forward()` function
    /// using the same states, as defined in the `DropoutDescriptor`.
    ///
    /// # Arguments
    ///
    /// * `dropout_descriptor` - previously created dropout descriptor.
    ///
    /// * `x_desc` - tensor descriptor for the operand.
    ///
    /// * `x_data` - data for the operand.
    ///
    /// * `y_desc` - tensor descriptor for the destination tensor.
    ///
    /// * `y_data` - data for the destination tensor.
    ///
    /// * `reserve_space` - user-allocated GPU memory used by this function. It is expected that the
    /// contents of reserveSpace does not change between `dropout_forward()` and
    /// `dropout_backward()` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `x_data` and `y_data` differs and if
    /// `reserve_space` is less than the value returned by `get_dropout_reserve_space_size`.
    pub fn dropout_forward<T>(
        &self,
        dropout_desc: &DropoutDescriptor<impl GpuBuffer<u8>>,
        x_desc: &TensorDescriptor<T>,
        x: &impl GpuBuffer<T>,
        y_desc: &TensorDescriptor<T>,
        y: &mut impl GpuBuffer<T>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
    {
        let x_data = x.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let y_data = y.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let reserve_space_ptr = reserve_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let reserve_space_size = reserve_space.len();

        unsafe {
            sys::cudnnDropoutForward(
                self.raw,
                dropout_desc.raw,
                x_desc.raw,
                x_data,
                y_desc.raw,
                y_data,
                reserve_space_ptr,
                reserve_space_size,
            )
            .into_result()
        }
    }

    /// This function performs backward dropout operation over `dy_data` returning results in
    /// `dx_data`.
    ///
    /// If during forward dropout operation value from `x_data` was propagated to `y_data` then
    /// during backward operation value from `dy_data` will be propagated to `dx_data`, otherwise,
    /// `dx_data` value will be set to 0.0.
    ///
    /// # Arguments
    ///
    /// * `dropout_descriptor` - previously created dropout descriptor.
    ///
    /// * `dx_desc` - tensor descriptor for the operand.
    ///
    /// * `dx` - data for the operand.
    ///
    /// * `dy_desc` - tensor descriptor for the destination tensor.
    ///
    /// * `dy` - data for the destination tensor.
    ///
    /// * `reserve_space` - user-allocated GPU memory used by this function. It is expected that the
    /// contents of reserveSpace does not change between `dropout_forward()` and
    /// `dropout_backward()` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `dx_data` and `dy_data` differs and if
    /// `reserve_space` is less than the value returned by `get_dropout_reserve_space_size`.
    pub fn dropout_backward<T>(
        &self,
        dropout_desc: &DropoutDescriptor<impl GpuBuffer<u8>>,
        dx_desc: &TensorDescriptor<T>,
        dx: &impl GpuBuffer<T>,
        dy_desc: &TensorDescriptor<T>,
        dy: &mut impl GpuBuffer<T>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
    {
        let dy_data = dx.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let dx_data = dy.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let reserve_space_ptr = reserve_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;
        let reserve_space_size = reserve_space.len();

        unsafe {
            sys::cudnnDropoutBackward(
                self.raw,
                dropout_desc.raw,
                dy_desc.raw,
                dy_data,
                dx_desc.raw,
                dx_data,
                reserve_space_ptr,
                reserve_space_size,
            )
            .into_result()
        }
    }
}
