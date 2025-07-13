#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

__global__ void voxel_pool_kernel(int B, int X, int Y, int Z, int N, int C, int n_intervals,
                                  const float* feats, const int* coords, const int* interval_starts,
                                  const int* interval_lengths, float* out) {
  // feats: (N, C)
  // coords: (N, 4), [bs_idx, x, y, z]
  // out: (B, X, Y, Z, C)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / C;
  int cur_c = idx % C;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_coords = coords + interval_start * 4;
  const float* cur_feats = feats + interval_start * C + cur_c;
  float* cur_out = out + cur_coords[0] * X * Y * Z * C +
                   cur_coords[1] * Y * Z * C + 
                   cur_coords[2] * Z * C +
                   cur_coords[3] * C + 
                   cur_c;
  float psum = 0;
  for(int i = 0; i < interval_length; i++){
    psum += cur_feats[i * C];
  }
  *cur_out = psum;
}

__global__ void voxel_pool_grad_kernel(int B, int X, int Y, int Z, int N, int C, int n_intervals,
                                       const float* out_grad, const int* coords, const int* interval_starts,
                                       const int* interval_lengths, float* feats_grad) {
  // out_grad: (B, X, Y, Z, C)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / C;
  int cur_c = idx % C;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_coords = coords + interval_start * 4;
  float* cur_feats_grad = feats_grad + interval_start * C + cur_c;
  const float* cur_out_grad = out_grad + cur_coords[0] * X * Y * Z * C +
                              cur_coords[1] * Y * Z * C +
                              cur_coords[2] * Z * C +
                              cur_coords[3] * C +
                              cur_c;
  for(int i = 0; i < interval_length; i++){
    cur_feats_grad[i * C] = *cur_out_grad;
  }
}

void voxel_pool(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* feats,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* out) {
  voxel_pool_kernel<<<DIVUP(n_intervals * C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
    B, X, Y, Z, N, C, n_intervals, feats, coords, interval_starts, interval_lengths, out
  );
}

void voxel_pool_grad(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* out_grad,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* feats_grad) {
  voxel_pool_grad_kernel<<<DIVUP(n_intervals * C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
    B, X, Y, Z, N, C, n_intervals, out_grad, coords, interval_starts, interval_lengths, feats_grad
  );
}
