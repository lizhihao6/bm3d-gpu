#include <torch/extension.h>

#include <vector>

#include "bm3d.hpp"
#include "params.hpp"

// CUDA forward declarations

torch::Tensor bm3d_cuda_forward(torch::Tensor input, BM3D& bm3d,
                                float& variance, bool twostep) {
  // Allocate images
  auto output = torch::zeros_like(input);

  int height = input.size(0);
  int width = input.size(1);

  // Launch BM3D
  try {
    bm3d.denoise_host_image(input.data_ptr<raw_int>(),
                            output.data_ptr<raw_int>(), width, height, 1,
                            &variance, twostep);
  } catch (std::exception& e) {
    std::cerr << "There was an error while processing image: " << std::endl
              << e.what() << std::endl;
    return output;
  }

  return output;
}

// C++ interface

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor bm3d_forward(torch::Tensor input, float& variance, bool twostep) {
  CHECK_INPUT(input);

  BM3D bm3d;
  bm3d.set_hard_params(19, 8, 16, 2500, 3, 2.7f);
  bm3d.set_wien_params(19, 8, 32, 400, 3);
  bm3d.set_verbose(false);

  return bm3d_cuda_forward(input, bm3d, variance, twostep);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bm3d_forward, "BM3D forward (CUDA)");
}