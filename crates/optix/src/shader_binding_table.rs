use crate::{
    context::DeviceContext,
    error::{Error, ToResult},
    optix_call,
    program_group::ProgramGroup,
    sys,
};

use cust::memory::{DSlice, DeviceCopy};
use cust_raw::CUdeviceptr;

type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone)]
pub struct SbtRecord<T>
where
    T: Copy,
{
    header: sys::SbtRecordHeader,
    data: T,
}

// impl<T> Copy for SbtRecord<T> where T: Copy {}

impl<T> SbtRecord<T>
where
    T: Copy,
{
    pub fn pack(data: T, program_group: &ProgramGroup) -> Result<SbtRecord<T>> {
        let mut rec = SbtRecord {
            header: sys::SbtRecordHeader::default(),
            data,
        };

        unsafe {
            Ok(optix_call!(optixSbtRecordPackHeader(
                program_group.raw,
                &mut rec as *mut _ as *mut std::os::raw::c_void,
            ))
            .map(|_| rec)?)
        }
    }
}

unsafe impl<T: DeviceCopy> DeviceCopy for SbtRecord<T> {}

#[repr(C)]
pub struct ShaderBindingTable {
    raygen_record: CUdeviceptr,
    exception_record: CUdeviceptr,
    miss_record_base: CUdeviceptr,
    miss_record_stride_in_bytes: u32,
    miss_record_count: u32,
    hitgroup_record_base: CUdeviceptr,
    hitgroup_record_stride_in_bytes: u32,
    hitgroup_record_count: u32,
    callables_record_base: CUdeviceptr,
    callables_record_stride_in_bytes: u32,
    callables_record_count: u32,
}

impl ShaderBindingTable {
    pub fn new<RG: DeviceCopy>(buf_raygen_record: &mut DSlice<SbtRecord<RG>>) -> Self {
        let raygen_record = buf_raygen_record.as_device_ptr().as_raw() as u64;
        ShaderBindingTable {
            raygen_record,
            exception_record: 0,
            miss_record_base: 0,
            miss_record_stride_in_bytes: 0,
            miss_record_count: 0,
            hitgroup_record_base: 0,
            hitgroup_record_stride_in_bytes: 0,
            hitgroup_record_count: 0,
            callables_record_base: 0,
            callables_record_stride_in_bytes: 0,
            callables_record_count: 0,
        }
    }

    pub fn build(self) -> sys::OptixShaderBindingTable {
        unsafe { std::mem::transmute::<ShaderBindingTable, sys::OptixShaderBindingTable>(self) }
    }

    pub fn exception<EX: DeviceCopy>(
        mut self,
        buf_exception_record: &mut DSlice<SbtRecord<EX>>,
    ) -> Self {
        if buf_exception_record.len() != 1 {
            panic!("SBT not psased single exception record",);
        }
        self.exception_record = buf_exception_record.as_device_ptr().as_raw() as u64;
        self
    }

    pub fn miss<MS: DeviceCopy>(mut self, buf_miss_records: &mut DSlice<SbtRecord<MS>>) -> Self {
        if buf_miss_records.len() == 0 {
            panic!("SBT passed empty miss records");
        }
        self.miss_record_base = buf_miss_records.as_device_ptr().as_raw() as u64;
        self.miss_record_stride_in_bytes = std::mem::size_of::<SbtRecord<MS>>() as u32;
        self.miss_record_count = buf_miss_records.len() as u32;
        self
    }

    pub fn hitgroup<HG: DeviceCopy>(
        mut self,
        buf_hitgroup_records: &mut DSlice<SbtRecord<HG>>,
    ) -> Self {
        if buf_hitgroup_records.len() == 0 {
            panic!("SBT passed empty hitgroup records");
        }
        self.hitgroup_record_base = buf_hitgroup_records.as_device_ptr().as_raw() as u64;
        self.hitgroup_record_stride_in_bytes = std::mem::size_of::<SbtRecord<HG>>() as u32;
        self.hitgroup_record_count = buf_hitgroup_records.len() as u32;
        self
    }

    pub fn callables<CL: DeviceCopy>(
        mut self,
        buf_callables_records: &mut DSlice<SbtRecord<CL>>,
    ) -> Self {
        if buf_callables_records.len() == 0 {
            panic!("SBT passed empty callables records");
        }
        self.callables_record_base = buf_callables_records.as_device_ptr().as_raw() as u64;
        self.callables_record_stride_in_bytes = std::mem::size_of::<SbtRecord<CL>>() as u32;
        self.callables_record_count = buf_callables_records.len() as u32;
        self
    }
}

// Sanity check that the size of this union we're defining matches the one in
// optix header so we don't get any nasty surprises
fn _size_check() {
    unsafe {
        std::mem::transmute::<ShaderBindingTable, [u8; sys::OptixShaderBindingTableSize]>(panic!());
    }
}
