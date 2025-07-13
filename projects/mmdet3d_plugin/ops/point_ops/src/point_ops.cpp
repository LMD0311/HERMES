#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

void group_inner_inds_launcher(int N, int M, int K, const long *inverse_inds, long *group_inds);

int group_inner_inds_wrapper(at::Tensor inverse_inds_tensor, at::Tensor group_inds_tensor) {
    CHECK_INPUT(inverse_inds_tensor);
    CHECK_INPUT(group_inds_tensor);

    int N = inverse_inds_tensor.size(0);
    int M = group_inds_tensor.size(0);
    int K = group_inds_tensor.size(1);

    const long *inverse_inds = inverse_inds_tensor.data_ptr<long>();
    long *group_inds = group_inds_tensor.data_ptr<long>();

    group_inner_inds_launcher(N, M, K, inverse_inds, group_inds);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_inner_inds_wrapper", &group_inner_inds_wrapper, "group_inner_inds_wrapper");
}