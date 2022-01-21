use crate::{const_assert, const_assert_eq};
use crate::{error::Error, optix_call, pipeline::ProgramGroup, sys};
use cust::memory::{DeviceCopy, DeviceSlice};

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

#[repr(transparent)]
pub struct ShaderBindingTable(pub(crate) sys::OptixShaderBindingTable);

impl ShaderBindingTable {
    pub fn new<RG: DeviceCopy>(buf_raygen_record: &DeviceSlice<SbtRecord<RG>>) -> Self {
        let raygen_record = buf_raygen_record.as_device_ptr().as_raw();
        ShaderBindingTable(sys::OptixShaderBindingTable {
            raygenRecord: raygen_record,
            exceptionRecord: 0,
            missRecordBase: 0,
            missRecordStrideInBytes: 0,
            missRecordCount: 0,
            hitgroupRecordBase: 0,
            hitgroupRecordStrideInBytes: 0,
            hitgroupRecordCount: 0,
            callablesRecordBase: 0,
            callablesRecordStrideInBytes: 0,
            callablesRecordCount: 0,
        })
    }

    pub fn exception<EX: DeviceCopy>(
        mut self,
        buf_exception_record: &DeviceSlice<SbtRecord<EX>>,
    ) -> Self {
        if buf_exception_record.len() != 1 {
            panic!("SBT not passed single exception record",);
        }
        self.0.exceptionRecord = buf_exception_record.as_device_ptr().as_raw();
        self
    }

    pub fn miss<MS: DeviceCopy>(mut self, buf_miss_records: &DeviceSlice<SbtRecord<MS>>) -> Self {
        if buf_miss_records.is_empty() {
            panic!("SBT passed empty miss records");
        }
        self.0.missRecordBase = buf_miss_records.as_device_ptr().as_raw();
        self.0.missRecordStrideInBytes = std::mem::size_of::<SbtRecord<MS>>() as u32;
        self.0.missRecordCount = buf_miss_records.len() as u32;
        self
    }

    pub fn hitgroup<HG: DeviceCopy>(
        mut self,
        buf_hitgroup_records: &DeviceSlice<SbtRecord<HG>>,
    ) -> Self {
        if buf_hitgroup_records.is_empty() {
            panic!("SBT passed empty hitgroup records");
        }
        self.0.hitgroupRecordBase = buf_hitgroup_records.as_device_ptr().as_raw();
        self.0.hitgroupRecordStrideInBytes = std::mem::size_of::<SbtRecord<HG>>() as u32;
        self.0.hitgroupRecordCount = buf_hitgroup_records.len() as u32;
        self
    }

    pub fn callables<CL: DeviceCopy>(
        mut self,
        buf_callables_records: &DeviceSlice<SbtRecord<CL>>,
    ) -> Self {
        if buf_callables_records.is_empty() {
            panic!("SBT passed empty callables records");
        }
        self.0.callablesRecordBase = buf_callables_records.as_device_ptr().as_raw();
        self.0.callablesRecordStrideInBytes = std::mem::size_of::<SbtRecord<CL>>() as u32;
        self.0.callablesRecordCount = buf_callables_records.len() as u32;
        self
    }
}

const_assert_eq!(
    std::mem::align_of::<ShaderBindingTable>(),
    std::mem::align_of::<sys::OptixShaderBindingTable>(),
);
const_assert_eq!(
    std::mem::size_of::<ShaderBindingTable>(),
    std::mem::size_of::<sys::OptixShaderBindingTable>()
);
