func_prefixes=("cublas.*" "cublasLt.*" "cublasXt.*")
var_prefixes=("CUBLAS.*" "CUBLASLT.*" "CUBLASXT.*")
iter=0
for f in cublas_v2.h cublasLt.h cublasXt.h; do
  bindgen "${CUDA_PATH}/include/$f" \
    --size_t-is-usize \
    --allowlist-type "${func_prefixes[$iter]}" \
    --allowlist-function "${func_prefixes[$iter]}" \
    --allowlist-var "${var_prefixes[$iter]}" \
    --no-layout-tests \
    --no-doc-comments \
    --default-enum-style rust \
    -- -I "${CUDA_PATH}/include" > src/${f%.*}.rs

  ((iter=iter+1))
done