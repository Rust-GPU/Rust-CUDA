DEPS_DIR="$HOME/deps"
OPTIX_VERSION="7.0"

echo "Used OptiX version: ${OPTIX_VERSION}"
mkdir -p ${DEPS_DIR}/optix/include
OPTIX_URL=https://developer.download.nvidia.com/redist/optix/v${OPTIX_VERSION}

for f in optix.h optix_device.h optix_function_table.h \
          optix_function_table_definition.h optix_host.h \
          optix_stack_size.h optix_stubs.h optix_types.h optix_7_device.h \
          optix_7_host.h optix_7_types.h \
          internal/optix_7_device_impl.h \
          internal/optix_7_device_impl_exception.h \
          internal/optix_7_device_impl_transformations.h
    do
    curl --retry 100 -m 120 --connect-timeout 30 \
        $OPTIX_URL/include/$f > $DEPS_DIR/optix/include/$f
done
OPTIX_ROOT=${DEPS_DIR}/optix
echo "OPTIX_ROOT=${OPTIX_ROOT}" >> $GITHUB_ENV
