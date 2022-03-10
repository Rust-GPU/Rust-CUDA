bindgen "${HOME}/local/include/cudnn.h" \
     --size_t-is-usize \
     --allowlist-type "cudnn.*" \
     --allowlist-function "cudnn.*" \
     --allowlist-var "CUDNN.*" \
     --no-layout-tests \
     --no-doc-comments \
     --default-enum-style rust \
     -- -I "/usr/local/cuda/include" > src/sys.rs
