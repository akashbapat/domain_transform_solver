#include "hfbs.h"

#include <iostream>
#include <limits>

#include "cudaArray2D.h"
#include "cudaArray3D.h"
#include "cudaSurface2D.h"
#include "cudaSurface3D.h"

#include "color_space.cuh"
#include "domain_transform_common_functions.h"
#include "domain_transform_common_kernels.cuh"
#include "error_types.h"

namespace domain_transform {
namespace {

// Helper kernel functions.
template <typename CudaType1, typename CudaType2>
__global__ void CopyVariable3d(const HFBSGridParams grid_params,
                               const CudaType1 src, CudaType2 dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 0 && y >= 0 && z >= 0 && x < grid_params.grid_x &&
      y < grid_params.grid_y && z < grid_params.grid_l) {
    dst.set(x, y, z, src.get(x, y, z));
  }
}

template <typename CudaType>
__global__ void Fill3dAndPad(const HFBSGridParams grid_params,
                             const typename CudaType::Scalar value,
                             CudaType dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 0 && y >= 0 && z >= 0 && x <= grid_params.grid_x &&
      y <= grid_params.grid_y && z <= grid_params.grid_l &&
      x < dst.get_width() && y < dst.get_height() && z < dst.get_depth()) {
    dst.set(x, y, z, value);
  }
}

template <typename CudaType>
void FillGrid(const HFBSGridParams grid_params,
              const typename CudaType::Scalar value, CudaType dst) {
  // Add one extra padding so that =-1 is also set.
  constexpr int kBlockSize3d = 8;
  const dim3 block_dim3d(kBlockSize3d, kBlockSize3d, kBlockSize3d);

  const dim3 grid_dim3d(
      dim3(std::ceil((grid_params.grid_x * 1.0f) / kBlockSize3d),
           std::ceil((grid_params.grid_y * 1.0f) / kBlockSize3d),
           std::ceil((grid_params.grid_l * 1.0f) / kBlockSize3d)));

  Fill3dAndPad<<<grid_dim3d, block_dim3d>>>(grid_params, value, dst);
}

}  // namespace

//  --------------------------Kernels used by solver---------------------------
//
// This kernel takes the confidence, target and the color image (assumed in luv
// space or lll space -- only first channel is used) and splats them into the
// bilateral grids b and c, which is the same as in the problem min 0.5 zAz -bz
// + c form Mazumdar et al.
template <typename CudaArrayType>
__global__ void SplatBAndC(const ImageDim image_dim,
                           const HFBSGridParams grid_params,
                           const cua::CudaArray2D<float> confidence,
                           const cua::CudaArray2D<float> target,
                           const cua::CudaSurface2D<uchar4> luv,
                           CudaArrayType b, CudaArrayType splat_c) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    const int grid_x = floorf(x * 1.0f / grid_params.sigma_x);
    const int grid_y = floorf(y * 1.0f / grid_params.sigma_y);

    const uchar4 luma = luv.get(x, y);
    const int grid_z = floorf(luma.x * 1.0f / grid_params.sigma_l);

    const float ct = confidence.get(x, y) * target.get(x, y);

    atomicAdd(b.ptr(grid_x, grid_y, grid_z), ct);
    atomicAdd(splat_c.ptr(grid_x, grid_y, grid_z), confidence.get(x, y));
  }
}

// This kernel populates all th buffers required to bistochastasize.
template <typename CudaArrayType>
__global__ void BistochastasizeSetup(const ImageDim image_dim,
                                     const HFBSGridParams grid_params,
                                     const cua::CudaArray2D<float> confidence,
                                     const cua::CudaSurface2D<uchar4> luv,
                                     CudaArrayType splat_sum,
                                     CudaArrayType inv_diag_A) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Fill splat_norm by ones, fill inv_diag_A with zeros, splat_sum by zeros
  // before calling this function.
  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    const int grid_x = floorf(x * 1.0f / grid_params.sigma_x);
    const int grid_y = floorf(y * 1.0f / grid_params.sigma_y);

    const uchar4 luma = luv.get(x, y);
    const int grid_z = floorf(luma.x * 1.0f / grid_params.sigma_l);

    atomicAdd(splat_sum.ptr(grid_x, grid_y, grid_z), 1);

    // Saving splat (c) into inv_diag_A
    atomicAdd(inv_diag_A.ptr(grid_x, grid_y, grid_z), confidence.get(x, y));
  }
}

// This kernel bistochastasizes the bilateral weights. This step corresponds to
// Eqn 9 in Mazumdar et al. This kernel assumes that BistochastasizeSetup() was
// previously called.
template <typename CudaArrayType, typename CudaSurfaceType>
__global__ void BistochastasizeOptimize(const ImageDim image_dim,
                                        const HFBSGridParams grid_params,
                                        const CudaArrayType splat_sum,
                                        const CudaSurfaceType splat_norm_src,
                                        CudaSurfaceType splat_norm_dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 0 && y >= 0 && z >= 0 && x < grid_params.grid_x &&
      y < grid_params.grid_y && z < grid_params.grid_l) {
    // This corresponds to diffusion operator D in the paper.
    const float diffuse_splat_norm =
        splat_norm_src.get(x - 1, y, z) + splat_norm_src.get(x + 1, y, z) +
        splat_norm_src.get(x, y - 1, z) + splat_norm_src.get(x, y + 1, z) +
        splat_norm_src.get(x, y, z - 1) + splat_norm_src.get(x, y, z + 1);

    const float splat_norm_val = splat_norm_src.get(x, y, z);

    // blurred_splat_norm corresponds to B in By = 2y + Dy in the paper.
    const float blurred_splat_norm = 2 * splat_norm_val + diffuse_splat_norm;

    // Value of epsilon = 0.00001.
    const float new_splat_norm =
        sqrt(splat_norm_val * (splat_sum.get(x, y, z) + 0.00001f) /
             (blurred_splat_norm + 0.00001f));

    splat_norm_dst.set(x, y, z, new_splat_norm);
  }
}

// Calcluates inverse of diagonal matrix A.
template <typename CudaArrayType, typename CudaSurfaceType>
__global__ void CalcInvDiagA(const ImageDim image_dim,
                             const HFBSGridParams grid_params,
                             const CudaArrayType splat_sum,
                             const CudaSurfaceType splat_norm,
                             CudaArrayType inv_diag_A) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 0 && y >= 0 && z >= 0 && x < grid_params.grid_x &&
      y < grid_params.grid_y && z < grid_params.grid_l) {
    // Calculate diagonal_A. c is already added.
    const float splat_norm_final = splat_norm.get(x, y, z);

    // (TODO:akashbapat) IS THIS A BUG OR NOT??
    const float splat_sum_bug = splat_sum.get(x, y, z);

    // WHAT I THINK IT SHOULD BE.
    //    const float splat_sum_bug = splat_norm_final*  (2*splat_norm_final +
    //   splat_norm.get(x-1, y, z) + splat_norm.get(x+1, y, z)
    // + splat_norm.get(x, y-1, z) + splat_norm.get(x, y+1, z)
    // + splat_norm.get(x, y, z-1) + splat_norm.get(x, y, z+1));

    const float diag_A_val =
        grid_params.lambda *
            (splat_sum_bug - 2 * splat_norm_final * splat_norm_final) +
        inv_diag_A.get(x, y, z);

// set inv_a value to inv_diag_A;

#define A_VERT_DIAG_MIN 1e-3

    inv_diag_A.set(x, y, z, 1.0f / max(A_VERT_DIAG_MIN, diag_A_val));

#undef A_VERT_DIAG_MIN
  }
}

// Solves the problem using Heavy-Ball gradient descent method. All the names in
// this function are the same as in the python code provided by the authors of
// Mazumdar et al.
template <typename CudaArrayType, typename CudaSurfaceType>
__global__ void SolveHFBS(const ImageDim image_dim,
                          const HFBSGridParams grid_params,
                          const CudaSurfaceType splat_norm,
                          const CudaArrayType b, const CudaArrayType inv_diag_A,
                          const CudaSurfaceType Y_c_src,
                          CudaSurfaceType Y_c_dst, CudaArrayType h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  // Y_c is pz in paper
  if (x >= 0 && y >= 0 && z >= 0 && x < grid_params.grid_x &&
      y < grid_params.grid_y && z < grid_params.grid_l) {
    const float diffused_Ysn_val =
        Y_c_src.get(x - 1, y, z) * splat_norm.get(x - 1, y, z) +
        Y_c_src.get(x + 1, y, z) * splat_norm.get(x + 1, y, z) +
        Y_c_src.get(x, y - 1, z) * splat_norm.get(x, y - 1, z) +
        Y_c_src.get(x, y + 1, z) * splat_norm.get(x, y + 1, z) +
        Y_c_src.get(x, y, z - 1) * splat_norm.get(x, y, z - 1) +
        Y_c_src.get(x, y, z + 1) * splat_norm.get(x, y, z + 1);

    const float qDqz_plus_b =
        (b.get(x, y, z) +
         grid_params.lambda * splat_norm.get(x, y, z) * diffused_Ysn_val) *
        inv_diag_A.get(x, y, z);

    const float Y_c_val = Y_c_src.get(x, y, z);
    const float grad = Y_c_val - qDqz_plus_b;
    const float h_val = h.get(x, y, z);

    // Solves for a robust gradient using Huber loss. If you dont want to use
    // Huber loss, set grad_clamped =  grad.
    const float grad_clamped = fabsf(grad) > 1.345 ? copysignf(1, grad) : grad;
    const float h_val_new = grid_params.beta * h_val + grad_clamped;
    h.set(x, y, z, h_val_new);

    Y_c_dst.set(x, y, z, Y_c_val - grid_params.alpha * h_val_new);
  }
}

// This slices the solution form the bilateral grid back to the pixel space. If
// kSliceTrilinear = true, this uses trilinear interpolation in the bilateral
// space which results into a much smoother solution.
template <bool kSliceTrilinear, typename CudaArrayType,
          typename CudaSurfaceType, typename CudaArraySolType>
__global__ void SliceSolution(const ImageDim image_dim,
                              const HFBSGridParams grid_params,
                              const cua::CudaSurface2D<uchar4> luv,
                              const CudaArrayType splat_sum,
                              const CudaSurfaceType Y_c,
                              CudaArraySolType optim_var) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    const float g_x = (x * 1.0f / grid_params.sigma_x);
    const float g_y = (y * 1.0f / grid_params.sigma_y);

    const uchar4 luma = luv.get(x, y);
    const float g_z = (luma.x * 1.0f / grid_params.sigma_l);

    // Do trilinear interpolation.
    // See https://en.wikipedia.org/wiki/Trilinear_interpolation  for the
    // corresponding formulae.

    // Splat sum takes care of empty grid voxels.
    const float g_x_0 = floorf(g_x);
    const float g_y_0 = floorf(g_y);
    const float g_z_0 = floorf(g_z);

    if (kSliceTrilinear) {
      const float g_x_1 = g_x_0 + 1;
      const float g_y_1 = g_y_0 + 1;
      const float g_z_1 = g_z_0 + 1;

      const float c00 = splat_sum.get(g_x_0, g_y_0, g_z_0) *
                            Y_c.get(g_x_0, g_y_0, g_z_0) * (1 - (g_x - g_x_0)) +
                        splat_sum.get(g_x_1, g_y_0, g_z_0) *
                            Y_c.get(g_x_1, g_y_0, g_z_0) * (g_x - g_x_0);

      const float c01 = splat_sum.get(g_x_0, g_y_0, g_z_1) *
                            Y_c.get(g_x_0, g_y_0, g_z_1) * (1 - (g_x - g_x_0)) +
                        splat_sum.get(g_x_1, g_y_0, g_z_1) *
                            Y_c.get(g_x_1, g_y_0, g_z_1) * (g_x - g_x_0);

      const float c10 = splat_sum.get(g_x_0, g_y_1, g_z_0) *
                            Y_c.get(g_x_0, g_y_1, g_z_0) * (1 - (g_x - g_x_0)) +
                        splat_sum.get(g_x_1, g_y_1, g_z_0) *
                            Y_c.get(g_x_1, g_y_1, g_z_0) * (g_x - g_x_0);

      const float c11 = splat_sum.get(g_x_0, g_y_1, g_z_1) *
                            Y_c.get(g_x_0, g_y_1, g_z_1) * (1 - (g_x - g_x_0)) +
                        splat_sum.get(g_x_1, g_y_1, g_z_1) *
                            Y_c.get(g_x_1, g_y_1, g_z_1) * (g_x - g_x_0);

      const float c0 = c00 * (1 - (g_y - g_y_0)) + c10 * (g_y - g_y_0);

      const float c1 = c01 * (1 - (g_y - g_y_0)) + c11 * (g_y - g_y_0);

      // Trilinear splat splat_sum.

      const float c00m =
          splat_sum.get(g_x_0, g_y_0, g_z_0) * (1 - (g_x - g_x_0)) +
          splat_sum.get(g_x_1, g_y_0, g_z_0) * (g_x - g_x_0);

      const float c01m =
          splat_sum.get(g_x_0, g_y_0, g_z_1) * (1 - (g_x - g_x_0)) +
          splat_sum.get(g_x_1, g_y_0, g_z_1) * (g_x - g_x_0);

      const float c10m =
          splat_sum.get(g_x_0, g_y_1, g_z_0) * (1 - (g_x - g_x_0)) +
          splat_sum.get(g_x_1, g_y_1, g_z_0) * (g_x - g_x_0);

      const float c11m =
          splat_sum.get(g_x_0, g_y_1, g_z_1) * (1 - (g_x - g_x_0)) +
          splat_sum.get(g_x_1, g_y_1, g_z_1) * (g_x - g_x_0);

      const float c0m = c00m * (1 - (g_y - g_y_0)) + c10m * (g_y - g_y_0);

      const float c1m = c01m * (1 - (g_y - g_y_0)) + c11m * (g_y - g_y_0);

      optim_var.set(x, y,
                    (c0 * (1 - (g_z - g_z_0)) + c1 * (g_z - g_z_0)) /
                        (c0m * (1 - (g_z - g_z_0)) + c1m * (g_z - g_z_0)));

    } else {
      optim_var.set(x, y, Y_c.get(g_x_0, g_y_0, g_z_0));
    }
  }
}

// Initalizes the solution in the gradient descent optimization using a
// laplacian blur if kUseLaplacianBlur = true, see the paper for formula of
// z_init.
template <bool kUseLaplacianBlur, typename CudaArrayType,
          typename CudaSurfaceType>
__global__ void InitZ(const HFBSGridParams grid_params, const CudaArrayType b,
                      const CudaArrayType inv_diag_A,
                      const CudaArrayType splat_c, CudaSurfaceType Y_c) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 0 && y >= 0 && z >= 0 && x < grid_params.grid_x &&
      y < grid_params.grid_y && z < grid_params.grid_l) {
    if (kUseLaplacianBlur) {
      Y_c.set(x, y, z, Y_c.get(x, y, z) / (splat_c.get(x, y, z) + 1e-7));

    } else {
      Y_c.set(x, y, z, b.get(x, y, z) * inv_diag_A.get(x, y, z));
    }
  }
}

// Performs a sweep in X (horizontal) direction to compute the laplacian
// smoothed value. Laplacian smoothing is the same as exponential smoothing.
template <typename CudaType1, typename CudaType2>
__global__ void LaplaceSweepX(const HFBSGridParams grid_params,
                              const CudaType1 src, CudaType2 dst) {
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (y >= 0 && z >= 0 && y < grid_params.grid_y && z < grid_params.grid_l) {
    float running_val = src.get(0, y, z);
    dst.set(0, y, z, running_val);
    for (int i = 1; i < grid_params.grid_x; i++) {
      running_val = src.get(i, y, z) + grid_params.neg_exp_beta * running_val;
      dst.set(i, y, z, running_val);
    }

    running_val = src.get(grid_params.grid_x - 1, y, z);
    for (int i = grid_params.grid_x - 2; i >= 0; i--) {
      running_val = grid_params.neg_exp_beta * running_val;

      dst.set(i, y, z, dst.get(i, y, z) + running_val);
      running_val = running_val + src.get(i, y, z);
    }
  }
}

template <typename CudaType1, typename CudaType2>
__global__ void LaplaceSweepY(const HFBSGridParams grid_params,
                              const CudaType1 src, CudaType2 dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 0 && z >= 0 && x < grid_params.grid_x && z < grid_params.grid_l) {
    float running_val = src.get(x, 0, z);
    dst.set(x, 0, z, running_val);
    for (int i = 1; i < grid_params.grid_y; i++) {
      running_val = src.get(x, i, z) + grid_params.neg_exp_beta * running_val;
      dst.set(x, i, z, running_val);
    }

    running_val = src.get(x, grid_params.grid_y - 1, z);
    for (int i = grid_params.grid_y - 2; i >= 0; i--) {
      running_val = grid_params.neg_exp_beta * running_val;

      dst.set(x, i, z, dst.get(x, i, z) + running_val);
      running_val = running_val + src.get(x, i, z);
    }
  }
}

template <typename CudaType1, typename CudaType2>
__global__ void LaplaceSweepZ(const HFBSGridParams grid_params,
                              const CudaType1 src, CudaType2 dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < grid_params.grid_x && y < grid_params.grid_y) {
    float running_val = src.get(x, y, 0);
    dst.set(x, y, 0, running_val);
    for (int i = 1; i < grid_params.grid_l; i++) {
      running_val = src.get(x, y, i) + grid_params.neg_exp_beta * running_val;
      dst.set(x, y, i, running_val);
    }

    running_val = src.get(x, y, grid_params.grid_l - 1);
    for (int i = grid_params.grid_l - 2; i >= 0; i--) {
      running_val = grid_params.neg_exp_beta * running_val;

      dst.set(x, y, i, dst.get(x, y, i) + running_val);
      running_val = running_val + src.get(x, y, i);
    }
  }
}

//  --------------------Class implementation starts-------------------------
//
// Holds all the Cuda memory buffers. I reuse some of the buffers to minimize
// the memory footprint, so be careful while changing the code.
struct HFBS::HFBSImpl {
  cua::CudaArray2D<float> confidence;
  cua::CudaArray2D<float> target;
  cua::CudaSurface2D<uchar4> color_image;

  // This holds the optimized result.
  cua::CudaArray2D<float> optim_var;

  // Most of the the buffer names correspond to the definitions in Mazumdar et
  // al.
  // This is the gradient estimate as explained in Algorithm 1 in Mazumdar et
  // al.
  cua::CudaArray3D<float> h;

  // This is the current solution as explained in Algorithm 1 in Mazumdar et al.
  cua::CudaSurface3D<float> z;

  // This is a bilateral grid b in Mazumdar et al in the problem
  // min 0.5 zAz -bz + c.
  cua::CudaArray3D<float> b;

  // This is a bilateral grid which stores the number of pixels mapping to a
  // vertex.
  cua::CudaArray3D<float> splat_sum;

  // This is a memory buffer which is reused in different ways.
  cua::CudaSurface3D<float> scratch_space;

  //
  cua::CudaSurface3D<float> splat_norm;

  // This is inverse of A, hence z =  inv_diag_A * b, see Mazumdar et al.
  cua::CudaArray3D<float> inv_diag_A;

  // This is a bilateral grid c in Mazumdar et al in the problem
  // min 0.5 zAz -bz + c.
  cua::CudaArray3D<float> splat_c;

  // The bilateral grid is allocated with maximum size defined during
  // construction. Only a portion of it is utilized according to the image size
  // during runtime.
  HFBSImpl(const int max_width, const int max_height, const int max_grid_x,
           const int max_grid_y, const int max_grid_l)
      : confidence(max_width, max_height),
        target(max_width, max_height),
        color_image(max_width, max_height),
        optim_var(max_width, max_height),

        h(max_grid_x, max_grid_y, max_grid_l),
        z(max_grid_x, max_grid_y, max_grid_l),
        b(max_grid_x, max_grid_y, max_grid_l),
        splat_sum(max_grid_x, max_grid_y, max_grid_l),
        scratch_space(max_grid_x, max_grid_y, max_grid_l),
        splat_norm(max_grid_x, max_grid_y, max_grid_l),
        inv_diag_A(max_grid_x, max_grid_y, max_grid_l),
        splat_c(max_grid_x, max_grid_y, max_grid_l) {
    // Check for successful allocation of memory.
    GPU_CHECK(cudaPeekAtLastError());
  }
};

void HFBS::SetFilterParams(const HFBSGridParams &grid_params) {
  grid_params_ = grid_params;
}

void HFBS::InitFrame(const COLOR_SPACE &color_space, void *color_image) {
  // Copy inputs to GPU.
  solver_impl_->color_image.Upload(image_dim_.width, image_dim_.height,
                                   reinterpret_cast<uchar4 *>(color_image));
  GPU_CHECK(cudaPeekAtLastError());

  switch (color_space) {
    case COLOR_SPACE::RGB: {
      break;
    }
    case COLOR_SPACE::YCbCr: {
      dim3 block_dim, grid_dim;
      ComputeBlockAndGridDim2D<false>(image_dim_, &block_dim, &grid_dim);

      RGB2YCbCr<<<grid_dim, block_dim>>>(image_dim_, solver_impl_->color_image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case COLOR_SPACE::YYY: {
      dim3 block_dim, grid_dim;
      ComputeBlockAndGridDim2D<false>(image_dim_, &block_dim, &grid_dim);

      RGB2YYY<<<grid_dim, block_dim>>>(image_dim_, solver_impl_->color_image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
  }
}

void HFBS::SetImageDim(const ImageDim &image_dim) {
  // Protect against larger image than whats provided in constructor.
  if (solver_impl_->color_image.get_height() < image_dim.height ||
      solver_impl_->color_image.get_width() < image_dim.width) {
    std::cerr
        << "Image size is more than the reserved Cuda memory in constructor."
        << std::endl;

  } else {
    image_dim_ = image_dim;
  }
}

void HFBS::InitFrame(const COLOR_SPACE &color_space, void *color_image,
                     float *target, float *confidence) {
  HFBS::InitFrame(color_space, color_image);

  solver_impl_->target.Upload(image_dim_.width, image_dim_.height, target);
  GPU_CHECK(cudaPeekAtLastError());
  if (confidence == nullptr) {
    solver_impl_->confidence.Fill(1.0f);
  } else {
    solver_impl_->confidence.Upload(image_dim_.width, image_dim_.height,
                                    confidence);
    GPU_CHECK(cudaPeekAtLastError());
  }
}

void HFBS::Download(const ImageType &image_type, float *image) const {
  switch (image_type) {
    case ImageType::TARGET: {
      solver_impl_->target.CopyTo(image_dim_.width, image_dim_.height, image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case ImageType::CONFIDENCE: {
      solver_impl_->confidence.CopyTo(image_dim_.width, image_dim_.height,
                                      image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case ImageType::OPTIMIZED_QUANTITY: {
      solver_impl_->optim_var.CopyTo(image_dim_.width, image_dim_.height,
                                     image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
  }
}

HFBS::HFBS(const ImageDim &max_image_dims, const int max_grid_x,
           const int max_grid_y, const int max_grid_l)
    : solver_impl_(new HFBS::HFBSImpl(max_image_dims.width,
                                      max_image_dims.height, max_grid_x,
                                      max_grid_y, max_grid_l)) {}

HFBS::~HFBS() {}

void HFBS::Optimize() {
  // Set the grid parameters.

  grid_params_.grid_x = std::ceil(image_dim_.width / grid_params_.sigma_x);
  grid_params_.grid_y = std::ceil(image_dim_.height / grid_params_.sigma_y);
  grid_params_.grid_l = std::ceil(255.0 / grid_params_.sigma_l);

  // Protect against large 3D bilateral space.
  if (grid_params_.grid_x > solver_impl_->h.get_width() ||
      grid_params_.grid_y > solver_impl_->h.get_height() ||
      grid_params_.grid_l > solver_impl_->h.get_depth()) {
    std::cerr << "Trying to use too large a bilateral space. Increase the blur "
                 "sigmas or allocate more memory via constructor."
              << std::endl;

    // Copy target to optimized variable so that if someone tries to download,
    // they should be equal.

    dim3 block_dim2d, grid_dim2d;
    ComputeBlockAndGridDim2D<false>(image_dim_, &block_dim2d, &grid_dim2d);

    CopyVariable<<<grid_dim2d, block_dim2d>>>(image_dim_, solver_impl_->target,
                                              solver_impl_->optim_var);

    return;
  }

  dim3 block_dim2d, grid_dim2d;
  ComputeBlockAndGridDim2D<false>(image_dim_, &block_dim2d, &grid_dim2d);

  FillGrid(grid_params_, 0.0f, solver_impl_->b);

  FillGrid(grid_params_, 0.0f, solver_impl_->splat_c);

  SplatBAndC<<<grid_dim2d, block_dim2d>>>(
      image_dim_, grid_params_, solver_impl_->confidence, solver_impl_->target,
      solver_impl_->color_image, solver_impl_->b, solver_impl_->splat_c);
  GPU_CHECK(cudaPeekAtLastError());

  // Bistochastasize the bilateral weights.
  // Fill splat_sum by ones, fill inv_diag_A with zeros.
  FillGrid(grid_params_, 0.0f, solver_impl_->inv_diag_A);
  FillGrid(grid_params_, 0.0f, solver_impl_->splat_sum);

  BistochastasizeSetup<<<grid_dim2d, block_dim2d>>>(
      image_dim_, grid_params_, solver_impl_->confidence,
      solver_impl_->color_image, solver_impl_->splat_sum,
      solver_impl_->inv_diag_A);
  GPU_CHECK(cudaPeekAtLastError());

  constexpr int kBlockSize3d = 8;
  const dim3 block_dim3d(kBlockSize3d, kBlockSize3d, kBlockSize3d);

  const dim3 grid_dim3d(
      dim3(std::ceil((grid_params_.grid_x * 1.0f) / kBlockSize3d),
           std::ceil((grid_params_.grid_y * 1.0f) / kBlockSize3d),
           std::ceil((grid_params_.grid_l * 1.0f) / kBlockSize3d)));

  FillGrid(grid_params_, 1.0f, solver_impl_->scratch_space);

  FillGrid(grid_params_, 0.0f, solver_impl_->splat_norm);

  cua::CudaSurface3D<float> splat_norm_src = solver_impl_->scratch_space;
  cua::CudaSurface3D<float> splat_norm_dst = solver_impl_->splat_norm;

  for (int bistoch = 0; bistoch < grid_params_.bistoch_iter_max; bistoch++) {
    BistochastasizeOptimize<<<grid_dim3d, block_dim3d>>>(
        image_dim_, grid_params_, solver_impl_->splat_sum, splat_norm_src,
        splat_norm_dst);

    GPU_CHECK(cudaPeekAtLastError());

    // Swap buffers.
    const cua::CudaSurface3D<float> tmp_swap_buffer = splat_norm_src;
    splat_norm_src = splat_norm_dst;
    splat_norm_dst = tmp_swap_buffer;
  }

  CopyVariable3d<<<grid_dim3d, block_dim3d>>>(grid_params_, splat_norm_src,
                                              solver_impl_->splat_norm);
  GPU_CHECK(cudaPeekAtLastError());

  CalcInvDiagA<<<grid_dim3d, block_dim3d>>>(
      image_dim_, grid_params_, solver_impl_->splat_sum,
      solver_impl_->splat_norm, solver_impl_->inv_diag_A);
  GPU_CHECK(cudaPeekAtLastError());

  FillGrid(grid_params_, 0.0f, solver_impl_->scratch_space);
  FillGrid(grid_params_, 0.0f, solver_impl_->h);
  FillGrid(grid_params_, 0.0f, solver_impl_->z);

  // Initialize the zinit smartly.
  // First do laplacian blur on c.
  LaplaceSweepX<<<dim3(1, grid_dim3d.y, grid_dim3d.z),
                  dim3(1, block_dim3d.y, block_dim3d.z)>>>(
      grid_params_, solver_impl_->splat_c, solver_impl_->scratch_space);
  GPU_CHECK(cudaPeekAtLastError());

  LaplaceSweepY<<<dim3(grid_dim3d.x, 1, grid_dim3d.z),
                  dim3(block_dim3d.x, 1, block_dim3d.z)>>>(
      grid_params_, solver_impl_->scratch_space, solver_impl_->splat_c);
  GPU_CHECK(cudaPeekAtLastError());

  LaplaceSweepZ<<<dim3(grid_dim3d.x, grid_dim3d.y, 1),
                  dim3(block_dim3d.x, block_dim3d.y, 1)>>>(
      grid_params_, solver_impl_->splat_c, solver_impl_->scratch_space);
  GPU_CHECK(cudaPeekAtLastError());

  CopyVariable3d<<<grid_dim3d, block_dim3d>>>(
      grid_params_, solver_impl_->scratch_space, solver_impl_->splat_c);

  // Do laplacian blur on b and write to z.
  LaplaceSweepX<<<dim3(1, grid_dim3d.y, grid_dim3d.z),
                  dim3(1, block_dim3d.y, block_dim3d.z)>>>(
      grid_params_, solver_impl_->b, solver_impl_->z);
  GPU_CHECK(cudaPeekAtLastError());

  LaplaceSweepY<<<dim3(grid_dim3d.x, 1, grid_dim3d.z),
                  dim3(block_dim3d.x, 1, block_dim3d.z)>>>(
      grid_params_, solver_impl_->z, solver_impl_->scratch_space);
  GPU_CHECK(cudaPeekAtLastError());

  LaplaceSweepZ<<<dim3(grid_dim3d.x, grid_dim3d.y, 1),
                  dim3(block_dim3d.x, block_dim3d.y, 1)>>>(
      grid_params_, solver_impl_->scratch_space, solver_impl_->z);
  GPU_CHECK(cudaPeekAtLastError());

  // Initialize z with the smart solution.
  InitZ<true><<<grid_dim3d, block_dim3d>>>(
      grid_params_, solver_impl_->b, solver_impl_->inv_diag_A,
      solver_impl_->splat_c, solver_impl_->z);
  GPU_CHECK(cudaPeekAtLastError());

  // Run the optimization.
  cua::CudaSurface3D<float> z_src = solver_impl_->z;
  cua::CudaSurface3D<float> z_dst = solver_impl_->scratch_space;
  for (int iter = 0; iter < grid_params_.optim_iter_max; iter++) {
    SolveHFBS<<<grid_dim3d, block_dim3d>>>(
        image_dim_, grid_params_, solver_impl_->splat_norm, solver_impl_->b,
        solver_impl_->inv_diag_A, z_src, z_dst, solver_impl_->h);
    GPU_CHECK(cudaPeekAtLastError());

    // Swap buffers.
    const cua::CudaSurface3D<float> tmp_swap_buffer = z_src;
    z_src = z_dst;
    z_dst = tmp_swap_buffer;
  }

  CopyVariable3d<<<grid_dim3d, block_dim3d>>>(grid_params_, z_src,
                                              solver_impl_->z);

  constexpr bool kSliceTrilinear = true;

  SliceSolution<kSliceTrilinear><<<grid_dim2d, block_dim2d>>>(
      image_dim_, grid_params_, solver_impl_->color_image,
      solver_impl_->splat_sum, solver_impl_->z, solver_impl_->optim_var);
  GPU_CHECK(cudaPeekAtLastError());
}

void HFBS::ClearAll() {
  solver_impl_->confidence.Fill(0.0f);
  solver_impl_->target.Fill(0.0f);
  solver_impl_->color_image.Fill(make_uchar4(0, 0, 0, 0));
  solver_impl_->optim_var.Fill(0.0f);

  solver_impl_->h.Fill(0.0f);
  solver_impl_->z.Fill(0.0f);
  solver_impl_->b.Fill(0.0f);
  solver_impl_->splat_sum.Fill(0.0f);
  solver_impl_->scratch_space.Fill(0.0f);
  solver_impl_->splat_norm.Fill(0.0f);
  solver_impl_->inv_diag_A.Fill(0.0f);
  solver_impl_->splat_c.Fill(0.0f);
}

}  // namespace domain_transform
