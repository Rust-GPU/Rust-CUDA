# Shader Binding Table

# Programming Guide...
<details>
<summary>Click here to expand programming guide</summary>

# Contents

- [Records](#records)
- [Layout](#layout)
- [Acceleration Structures](#acceleration-structures)
    - [SBT Instance Offset](#sbt-instance-offset)
    - [SBT Geometry-AS Offset](#sbt-geometry-as-offset)
    - [SBT Trace Offset](#sbt-trace-offset)
    - [SBT Trace Stride](#sbt-trace-stride)
    - [Example SBT For a Scene](#example-sbt-for-a-scene)
- [SBT Record Access on Device](#sbt-record-access-on-device)

The shader binding table (SBT) is an array that contains information about the location of programs and their parameters. The SBT resides in device memory and is managed by the application.

The shader binding table can be complex to get your head around. In addition to this documentation, you might also enjoy reading Will Usher's [*The RTX Shader Binding Table Three Ways*](https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways)

## Records
A record is an array element of the SBT that consists of a header and a data block. The header content is opaque to the application, containing information accessed by traversal execution to identify and invoke programs. 

Rather than pack the records manually as in the C API, the Rust API instead gives you a generic [`SbtRecord<T>`] type that you can specialize to supply your data to the SBT:

```
#[derive(Copy, Clone, Default, DeviceCopy)]
struct HitgroupSbtData {
    object_id: u32,
}
type HitgroupRecord = SbtRecord<HitgroupSbtData>;

// Pack the object ids into the record for each object. In a real application 
// you would supply pointers to device memory containing vertex attributes 
// such as smoothed normals, texture coordinates etc.
let rec_hitgroup: Vec<_> = (0..num_objects)
    .map(|i| {
        let object_type = 0;
        let rec = HitgroupRecord::pack(
            HitgroupSbtData { object_id: i },
            &pg_hitgroup[object_type],
        )
        .expect("failed to pack hitgroup record");
        rec
    })
    .collect();


```

The data section of an [`SbtRecord`] can be accessed on the device using the `optixGetSbtDataPointer()` device function. 

## Layout

A shader binding table is split into five sections, where each section represents a unique program group type:

 <table>
 <tr><th>Group</th><th>Program Types in Group</th>
 <tr><td>Ray Generation</td><td><code>ray-generation</code></td></tr>
 <tr><td>Exception</td><td><code>exception</code></td></tr>
 <tr><td>Miss</td><td><code>miss</code></td></tr>
 <tr><td>Hit</td><td><code>closest-hit, any-hit, intersection</code></td></tr>
 <tr><td>Callable</td><td><code>direct-callable, continuation-callable</code></td></tr>
 </table>

 See also [Program Group Creation](crate::pipeline)

 The [`ShaderBindingTable`] is created by passing [`DeviceBuffer`]s of [`SbtRecord`]s to the constructor:

 ```
let mut buf_raygen = DeviceBuffer::from_slice(&rec_raygen)?;
let mut buf_miss = DeviceBuffer::from_slice(&rec_miss)?;
let mut buf_hitgroup = DeviceBuffer::from_slice(&rec_hitgroup)?;

let sbt = ShaderBindingTable::new(&mut buf_raygen)
    .miss(&mut buf_miss)
    .hitgroup(&mut buf_hitgroup)
    .build();
 ```
The [`SbtRecord`]s buffers are assumed to be densely, packed and the [`SbtRecord`] itself is correctly aligned to 16 bytes.

The index to records in the shader binding table is used in different ways for the miss, hit, and callables groups:

* *Miss* - Miss programs are selected for every optixTrace call using the missSBTIndex parameter.
* *Callables* - Callables take the index as a parameter and call the direct-callable when invoking optixDirectCall and continuation-callable when invoking optixContinuationCall.
* *Any-hit, closest-hit, intersection* - The computation of the index for the hit group (intersection, any-hit, closest-hit) is done during traversal. See [Acceleration structures](#acceleration-structures) for more detail.


## Acceleration Structures

The selection of the SBT hit group record for the instance is slightly more involved to allow for a number of use cases such as the implementation of different ray types. The SBT record index `sbt_index` is determined by the following index calculation during traversal:

```text
sbt-index =
    sbt-instance-offset
    + (sbt-geometry-acceleration-structure-index * sbt-stride-from-trace-call)
    + sbt-offset-from-trace-call
```

The index calculation depends upon the following SBT indices and offsets:

* Instance offset
* Geometry acceleration structure index
* Trace offset
* Trace stride

### SBT Instance Offset

Instance acceleration structure instances (type [`Instance`]) store an SBT offset that is applied during traversal. This is zero for single geometry-AS traversable because there is no corresponding instance-AS to hold the value. (See “Traversal of a single geometry acceleration structure”.) This value is limited to 24 bits.

### SBT Geometry-AS Index

Each geometry acceleration structure build input references at least one SBT record. The first SBT geometry acceleration structure index for each geometry acceleration structure build input is the prefix sum of the number of SBT records. Therefore, the computed SBT geometry acceleration structure index is dependent on the order of the build inputs.

The following example demonstrates a geometry acceleration structure with three build inputs. Each build input references one SBT record by specifying `num_sbt_records=1`. When intersecting geometry at trace time, the SBT geometry acceleration structure index used to compute the `sbt_index` to select the hit group record will be organized as follows:

<table>
<tr><th>SBT Geometry-AS Index</th><th>0</th><th>1</th><th>2</th></tr>
<tr><td rowspan=3>Geometry-AS build input</td><td><code>build_input[0]</code></td><td></td><td></td></r>
<tr><td></td><td><code>built_input[1]</code></td><td></td></r>
<tr><td></td><td></td><td><code>built_input[2]</code></td></r>
</table>

In this simple example, the index for the build input equals the SBT geometry acceleration structure index. Hence, whenever a primitive from “Build input [1]” is intersected, the SBT geometry acceleration structure index is one.

When a single build input references multiple SBT records (for example, to support multiple materials per geometry), the mapping corresponds to the prefix sum over the number of referenced SBT records.

For example, consider three build inputs where the first build input references four SBT records, the second references one SBT record, and the last references two SBT records:

<table>
<tr><th>SBT Geometry-AS Index</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th></tr>
<tr><td rowspan=3>Geometry-AS build input</td><td><code>build_input[0] num=4</code></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td><code>build_input[1] num=1</code></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td><code>build_input[2] offset=2</code></td><td></td></tr>
</table>

These three build inputs result in the following possible SBT geometry acceleration structure indices when intersecting the corresponding geometry acceleration structure build input:

* One index in the range of [0,3] if a primitive from `build_input[0]` is intersected
* Four if a primitive from `build_input[1]` is intersected
* One index in the range of [5,6] if a primitive from `build_input[2]` is intersected

The per-primitive SBT index offsets, as specified by using `sbt_index_offset_buffer`, are local to the build input. Hence, per-primitive offsets in the range [0,3] for the build input 0 and in the range [0,1] for the last build input, map to the SBT geometry acceleration structure index as follows:

<table>
<tr>
<th>SBT Geometry-AS Index</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th>
</tr>
<tr>
<td rowspan=4><code>build_input[0].sbt_index_offset:</code></td><td>[0]</td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>
<tr>
<td></td><td>[1]</td><td></td><td></td><td></td><td></td><td></td>
</tr>
<tr>
<td></td><td></td><td>[2]</td><td></td><td></td><td></td><td></td>
</tr>
<tr>
<td></td><td></td><td></td><td>[3]</td><td></td><td></td><td></td>
</tr>
<tr>
<td><code>build_input[1].sbt_index_offset=None</code></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>
<tr>
<td rowspan=2><code>build_input[1].sbt_index_offset:</code></td><td></td><td></td><td></td><td></td><td></td><td>[0]</td><td></td>
</tr>
<tr>
<td></td><td></td><td></td><td></td><td></td><td></td><td>[1]</td>
</tr>
</table>

Because `build_input[1]` references a single SBT record, a `sbt_index_offset_buffer` does not need to be specified for the geometry acceleration structure build. See “Acceleration structures”.

### SBT Trace Offset

The `optixTrace` function takes the parameter `SBToffset`, allowing for an SBT access shift for this specific ray. It is required to implement different ray types, i.e. the offset is the index of the ray type.

### SBT Trace Stride

The parameter `SBTstride`, defined as an index offset, is multiplied by `optixTrace` with the SBT geometry acceleration structure index. It is required to implement different ray types, i.e. the stride is the number of ray types.

### Example SBT For a Scene

In this example, a shader binding table implements the program selection for a simple scene containing one instance acceleration structure and two instances of the same geometry acceleration structure, where the geometry acceleration structure has two build inputs:

![Structure of a simple scene](scene_graph)

The first build input references a single SBT record, while the second one references two SBT records. There are two ray types: one for forward path tracing and one for shadow rays (next event estimation). The two instances of the geometry acceleration structure have different transforms and SBT offsets to allow for material variation in each instance of the same geometry acceleration structure. Therefore, the SBT needs to hold two miss records and 12 hit group records (three for the geometry acceleration structure, ×2 for the ray types, ×2 for the two instances in the instance acceleration structure).

![Example SBT](example_sbt)

To trace a ray of type 0 (for example, for path tracing):

```
optixTrace(IAS_handle,
    ray_org, ray_dir,
    tmin, tmax, time, 
    visMask, rayFlags,
    0, // sbtOffset
    2, // sbtStride
    0, // missSBTIndex 
    rayPayload0, ...);
```
Shadow rays need to pass in an adjusted `sbtOffset` as well as `missSBTIndex`:

```
optixTrace(IAS_handle,
    ray_org, ray_dir,
    tmin, tmax, time, 
    visMask, rayFlags,
    1, // sbtOffset
    2, // sbtStride
    1, // missSBTIndex 
    rayPayload0, ...);
```

Program groups of different types (ray generation, miss, intersection, and so on) do not need to be adjacent to each other as shown in the example. The pointer to the first SBT record of each program group type is passed to [`launch()`](crate::launch), as described previously, which allows for arbitrary spacing in the SBT between the records of different program group types.

### SBT Record Access on Device

To access the SBT data section of the currently running program, request its pointer by using an API function:
```text
CUdeviceptr optixGetSbtDataPointer();
```
Typically, this pointer is cast to a pointer that represents the layout of the data section. For example, for a closest hit program, the application gets access to the data associated with the SBT record that was used to invoke that closest hit program:

```text
struct CHData {
    int meshIdx; // Triangle mesh build input index
    float3 base_color;
};

CHData* material_info = (CHData*)optixGetSbtDataPointer();
```
The program is encouraged to rely on the alignment constraints of the SBT data section to read this data efficiently.

</details>


 [`Instance`]: crate::acceleration::Instance
 [`SbtRecord`]: crate::shader_binding_table::SbtRecord
 [`SbtRecord<T>`]: crate::shader_binding_table::SbtRecord
 [`ShaderBindingTable`]: crate::shader_binding_table::ShaderBindingTable
 [`DeviceBuffer`]: cust::memory::DeviceBuffer


