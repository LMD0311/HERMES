#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations
void voxel_pool(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* feats,
    const int* coords, const int* interval_starts, const int* interval_lengths, float* out);

void voxel_pool_grad(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* out_grad,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* feats_grad);


at::Tensor voxel_pool_forward(const at::Tensor _feats, const at::Tensor _coords, const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts, int B, int X, int Y, int Z
) {
  int N = _feats.size(0);
  int C = _feats.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feats));
  const float* feats = _feats.data_ptr<float>();
  const int* coords = _coords.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  
  auto options = torch::TensorOptions().dtype(_feats.dtype()).device(_feats.device());
  at::Tensor _out = torch::zeros({B, X, Y, Z, C}, options);
  float* out = _out.data_ptr<float>();

  voxel_pool(B, X, Y, Z, N, C, n_intervals,
    feats, coords, interval_starts, interval_lengths, out
  );
  
  return _out;
}

at::Tensor voxel_pool_backward(const at::Tensor _out_grad, const at::Tensor _coords, const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts, int B, int X, int Y, int Z
) {
  int N = _coords.size(0);
  int C = _out_grad.size(-1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* coords = _coords.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _feats_grad = torch::zeros({N, C}, options);
  float* feats_grad = _feats_grad.data_ptr<float>();

  voxel_pool_grad(
    B, X, Y, Z, N, C, n_intervals, out_grad,
    coords, interval_starts, interval_lengths, feats_grad
  );
  
  return _feats_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_pool_forward", &voxel_pool_forward, "voxel_pool_forward");
  m.def("voxel_pool_backward", &voxel_pool_backward, "voxel_pool_backward");
}
