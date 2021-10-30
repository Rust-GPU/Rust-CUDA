#include <optix.h>
#include <optix_host.h>

static const size_t OptixSbtRecordHeaderSize = OPTIX_SBT_RECORD_HEADER_SIZE;
static const size_t OptixSbtRecordAlignment = OPTIX_SBT_RECORD_ALIGNMENT;
static const size_t OptixAccelBufferByteAlignment =
    OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
static const size_t OptixInstanceByteAlignment = OPTIX_INSTANCE_BYTE_ALIGNMENT;
static const size_t OptixAabbBufferByteAlignment =
    OPTIX_AABB_BUFFER_BYTE_ALIGNMENT;
static const size_t OptixGeometryTransformByteAlignment =
    OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT;
static const size_t OptixTransformByteAlignment =
    OPTIX_TRANSFORM_BYTE_ALIGNMENT;

static const size_t OptixVersion = OPTIX_VERSION;

static const size_t OptixBuildInputSize = sizeof(OptixBuildInput);
static const size_t OptixShaderBindingTableSize = sizeof(OptixShaderBindingTable);

/**
 * <div rustbindgen replaces="OptixGeometryFlags"></div>
 */
enum GeometryFlags {
    None = OPTIX_GEOMETRY_FLAG_NONE,
    DisableAnyHit = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    RequireSingleAnyHitCall = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
};
