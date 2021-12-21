//! Scalar and Vector-based operations

use crate::{
    context::CublasContext,
    error::{Error, ToResult},
    raw::Level1,
    BlasDatatype,
};
use cust::memory::{GpuBox, GpuBuffer};
use cust::stream::Stream;

type Result<T = (), E = Error> = std::result::Result<T, E>;

fn check_stride<T: BlasDatatype>(x: &impl GpuBuffer<T>, n: usize, stride: Option<usize>) {
    let raw_len = x.len();
    let needed_len = n * stride.unwrap_or(1);
    assert!(
        raw_len >= needed_len,
        "Buffer is not long enough! required_len is {} ({} stride * {} n) but the buffer length is {}",
        needed_len,
        stride.unwrap_or(1),
        n,
        raw_len
    );
}

/// Scalar and Vector-based operations such as `min`, `max`, `axpy`, `copy`, `dot`, `nrm2`, `rot`, `rotg`, `rotm`, `rotmg`, `scal`, and `swap`.

impl CublasContext {
    /// Same as [`CublasContext::amin`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not long enough for the stride and length requested.
    pub fn amin_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        n: usize,
        stride: Option<usize>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        check_stride(x, n, stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::amin(
                ctx.raw,
                x.len() as i32,
                x.as_device_ptr().as_raw(),
                stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_raw_mut(),
            )
            .to_result()?)
        })
    }

    /// Finds the index of the smallest element inside of the GPU buffer by absolute value, writing the resulting
    /// index into `result`. The index is 1-based, not 0-based.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let data = [0.0f32, 1.0, 0.5, 5.0].as_dbuf()?;
    /// let mut result = DeviceBox::new(&0)?;
    ///
    /// ctx.amin(&stream, &data, &mut result)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(result.as_host_value()?, 1);
    /// drop((result, data, ctx, stream));
    /// # Ok(())
    /// # }
    /// ```
    pub fn amin<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        self.amin_strided(stream, x, x.len(), None, result)
    }

    /// Same as [`CublasContext::amax`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not long enough for the stride and length requested.
    pub fn amax_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        n: usize,
        stride: Option<usize>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        check_stride(x, n, stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::amax(
                ctx.raw,
                x.len() as i32,
                x.as_device_ptr().as_raw(),
                stride.unwrap_or(1) as i32,
                result.as_device_ptr().as_raw_mut(),
            )
            .to_result()?)
        })
    }

    /// Finds the index of the smallest element inside of the GPU buffer by absolute value, writing the resulting
    /// index into `result`. The index is 1-based, not 0-based.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let data = [0.0f32, 1.0, 0.5, 5.0].as_dbuf()?;
    /// let mut result = DeviceBox::new(&0)?;
    ///
    /// ctx.amax(&stream, &data, &mut result)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(result.as_host_value()?, 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn amax<T: Level1>(
        &mut self,
        stream: &Stream,
        x: &impl GpuBuffer<T>,
        result: &mut impl GpuBox<i32>,
    ) -> Result {
        self.amax_strided(stream, x, x.len(), None, result)
    }

    /// Same as [`CublasContext::axpy`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn axpy_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        alpha: &impl GpuBox<T>,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::axpy(
                ctx.raw,
                x.len() as i32,
                alpha.as_device_ptr().as_raw(),
                x.as_device_ptr().as_raw(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_raw_mut(),
                y_stride.unwrap_or(1) as i32,
            )
            .to_result()?)
        })
    }

    /// Multiplies `n` elements in `x` by `alpha`, then adds the result to `y`, overwriting
    /// `y` with the result.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` are not long enough for the requested length `n`.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let alpha = DeviceBox::new(&2.0)?;
    /// let x = [1.0, 2.0, 3.0, 4.0].as_dbuf()?;
    /// let mut y = [1.0; 4].as_dbuf()?;
    ///
    /// ctx.axpy(&stream, &alpha, x.len(), &x, &mut y)?;
    ///
    /// stream.synchronize()?;
    ///
    /// let result = y.as_host_vec()?;
    /// assert_eq!(&result, &[3.0, 5.0, 7.0, 9.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn axpy<T: Level1>(
        &mut self,
        stream: &Stream,
        alpha: &impl GpuBox<T>,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result {
        self.axpy_strided(stream, alpha, n, x, None, y, None)
    }

    /// Same as [`CublasContext::copy`] but with an explicit stride.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are not long enough for the stride and length requested.
    pub fn copy_strided<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        x_stride: Option<usize>,
        y: &mut impl GpuBuffer<T>,
        y_stride: Option<usize>,
    ) -> Result {
        check_stride(x, n, x_stride);
        check_stride(y, n, y_stride);

        self.with_stream(stream, |ctx| unsafe {
            Ok(T::copy(
                ctx.raw,
                x.len() as i32,
                x.as_device_ptr().as_raw(),
                x_stride.unwrap_or(1) as i32,
                y.as_device_ptr().as_raw_mut(),
                y_stride.unwrap_or(1) as i32,
            )
            .to_result()?)
        })
    }

    /// Copies `n` elements from `x` into `y`, overriding any previous data inside `y`.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` are not large enough for the requested amount of elements.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _a = cust::quick_init()?;
    /// # use blastoff::CublasContext;
    /// # use cust::prelude::*;
    /// # use cust::memory::DeviceBox;
    /// # use cust::util::SliceExt;
    /// # let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    /// let mut ctx = CublasContext::new()?;
    /// let x = [1.0f32, 2.0, 3.0, 4.0].as_dbuf()?;
    /// let mut y = [0.0; 4].as_dbuf()?;
    ///
    /// ctx.copy(&stream, x.len(), &x, &mut y)?;
    ///
    /// stream.synchronize()?;
    ///
    /// assert_eq!(x.as_host_vec()?, y.as_host_vec()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy<T: Level1>(
        &mut self,
        stream: &Stream,
        n: usize,
        x: &impl GpuBuffer<T>,
        y: &mut impl GpuBuffer<T>,
    ) -> Result {
        self.copy_strided(stream, n, x, None, y, None)
    }
}
