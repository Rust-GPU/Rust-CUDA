extern "C" __constant__ int my_constant = 314;

extern "C" __global__ void sum(const float *x, const float *y, float *out, int count)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
  {
    out[i] = x[i] + y[i];
  }
}