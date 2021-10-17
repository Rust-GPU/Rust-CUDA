use crate::{
    error::CudaResult,
    memory::{
        array::{ArrayObject, ArrayPrimitive},
        DBox, DeviceCopy,
    },
    prelude::DBuffer,
    surface::Surface,
    texture::Texture,
};

pub trait DeviceCopyExt: DeviceCopy {
    /// Makes a new [`DBox`] from this value.
    fn as_dbox(&self) -> CudaResult<DBox<Self>> {
        DBox::new(self)
    }
}

impl<T: DeviceCopy> DeviceCopyExt for T {}

/// Utilities for slices and slice-like things such as arrays.
pub trait SliceExt<T: DeviceCopy> {
    /// Allocate memory on the GPU and convert this slice into a DBuffer.
    fn as_dbuf(&self) -> CudaResult<DBuffer<T>>;
    fn as_1d_array(&self) -> CudaResult<ArrayObject>
    where
        T: ArrayPrimitive;

    fn as_2d_array(&self, width: usize, height: usize) -> CudaResult<ArrayObject>
    where
        T: ArrayPrimitive;

    fn as_1d_texture(&self) -> CudaResult<Texture>
    where
        T: ArrayPrimitive,
    {
        Texture::from_array(self.as_1d_array()?)
    }

    fn as_2d_texture(&self, width: usize, height: usize) -> CudaResult<Texture>
    where
        T: ArrayPrimitive,
    {
        Texture::from_array(self.as_2d_array(width, height)?)
    }

    fn as_1d_surface(&self) -> CudaResult<Surface>
    where
        T: ArrayPrimitive,
    {
        Surface::from_array(self.as_1d_array()?)
    }

    fn as_2d_surface(&self, width: usize, height: usize) -> CudaResult<Surface>
    where
        T: ArrayPrimitive,
    {
        Surface::from_array(self.as_2d_array(width, height)?)
    }
}

impl<T: DeviceCopy> SliceExt<T> for &[T] {
    fn as_dbuf(&self) -> CudaResult<DBuffer<T>> {
        DBuffer::from_slice(*self)
    }

    fn as_1d_array(&self) -> CudaResult<ArrayObject>
    where
        T: ArrayPrimitive,
    {
        let mut arr = ArrayObject::new_1d(self.len(), T::array_format(), 1)?;
        arr.copy_from(self)?;
        Ok(arr)
    }

    fn as_2d_array(&self, width: usize, height: usize) -> CudaResult<ArrayObject>
    where
        T: ArrayPrimitive,
    {
        let mut arr = ArrayObject::new_2d([width, height], T::array_format(), 1)?;
        arr.copy_from(self)?;
        Ok(arr)
    }
}

impl<T: DeviceCopy, const N: usize> SliceExt<T> for [T; N] {
    fn as_dbuf(&self) -> CudaResult<DBuffer<T>> {
        DBuffer::from_slice(self)
    }

    fn as_1d_array(&self) -> CudaResult<ArrayObject>
    where
        T: ArrayPrimitive,
    {
        let mut arr = ArrayObject::new_1d(self.len(), T::array_format(), 1)?;
        arr.copy_from(self)?;
        Ok(arr)
    }

    fn as_2d_array(&self, width: usize, height: usize) -> CudaResult<ArrayObject>
    where
        T: ArrayPrimitive,
    {
        let mut arr = ArrayObject::new_2d([width, height], T::array_format(), 1)?;
        arr.copy_from(self)?;
        Ok(arr)
    }
}
