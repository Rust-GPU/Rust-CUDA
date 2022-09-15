#!/bin/bash
set -exu

if [ -z ${CUDA_PATH} ]; then
  echo 'env var ${CUDA_PATH} must be defined, and must point to the root directory of the target Cuda installation'
  exit 1
fi

bindgen \
  --allowlist-type="^CU.*" \
  --allowlist-type="^cuuint(32|64)_t" \
  --allowlist-type="^cudaError_enum" \
  --allowlist-type="^cu.*Complex$" \
  --allowlist-type="^cuda.*" \
  --allowlist-type="^libraryPropertyType.*" \
  --allowlist-var="^CU.*" \
  --allowlist-function="^cu.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  wrapper.h \
  -- \
  -I${CUDA_PATH}/include \
  > src/cuda.rs
