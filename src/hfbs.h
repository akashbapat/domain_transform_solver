#ifndef HFBS_H_
#define HFBS_H_

#include <cuda_runtime.h>
#include <memory>

#include "domain_transform_common.h"

namespace domain_transform {
// Implements the method HFBS by Mazumdar et al., A Hardware-Friendly Bilateral
// Solver for Real-Time VirtualReality Video. This code was written by refering
// the paper and the open-source python code released by the author with MIT
// license at
// https://github.com/uwsampa/hfbs/tree/696c0ba50ca35ea885a662bfe3dc8927914692c7
//
// HFBSGridParams specifies the parameters for the bilateral space grid and the
// for the solver itself.
struct HFBSGridParams {
  // Sigma in x (horizontal) direction, do not use sigma less than 1.
  float sigma_x = 8.0f;

  // Sigma in y (vertical) direction, do not use sigma less than 1.
  float sigma_y = 8.0f;

  // Sigma in luma direction, do not use sigma less than 1.
  float sigma_l = 4.0f;

  // Optimization parameters.
  //
  // Lambda value for the spatial smoothness term, higher lambda means more
  // smoothing.
  float lambda = 8.0f;

  // These parameters used for Heavy-Ball gradient descent method.
  // alpha is the step size, see Algorithm 1 in the paper Mazumdar et al.
  float alpha = 1.0f;

  // beta is the momentum, see Algorithm 1 in the paper Mazumdar et al.
  float beta = 0.9f;

  // Number of iterations used for bistochastization, higher the better result.
  // Although the paper says only one iteration is enough, in practice multiple
  // iterations turns out to be more accurate.
  int bistoch_iter_max = 20;

  // Number of iterations used for optimization algorithm, higher the better
  // result.
  int optim_iter_max = 140;

  // neg_exp_beta = exp(-(|tau_x|+|tau_y|+|tau_z|)/sigma_b) used for laplacian
  // blurring for a better initialization to help reduce number of iterations.
  float neg_exp_beta = 0.9;

  // These parameters define the grid size of the bilateral space and are
  // automatically computed using the blur sigmas and image size. Do not set by
  // yourself.
  int grid_x = 1;
  int grid_y = 1;
  int grid_l = 1;
};

// This class implements the HFBS algorithm.
// The typical usage is as follows:
//
//  domain_transform::HFBSGridParams solver_options;
//  solver_options.sigma_x = 2;
//  solver_options.sigma_y = 2;
//  solver_options.sigma_l = 2;
//
//  solver_options.optim_iter_max = 125;
//  solver_options.bistoch_iter_max = 20;
//  solver_options.lambda = 1.5f;
//
//  // Make sure to allocate the CPU memory with same image dimensions.
//  cv::Mat confidence, target, rgba_image, result;
//
//  domain_transform::ImageDim image_dim;
//  image_dim.width = target.cols;
//  image_dim.height = target.rows;
//
//  domain_transform::HFBS solver(image_dim, 1536, 1536, 128);
//  solver.SetImageDim(image_dim);
//  solver.SetFilterParams(solver_options);
//
//  solver.ClearAll();
//
//  // HFBS needs YCbCr.
//  solver.InitFrame(domain_transform::COLOR_SPACE::YCbCr, rgba_image.data,
//                   reinterpret_cast<float*>(target.data),
//                   reinterpret_cast<float*>(confidence.data));
//
//  solver.Optimize();
//
//  targetsolver.Download(domain_transform::ImageType::OPTIMIZED_QUANTITY,
//                reinterpret_cast<float*>(result.data));
//
//
//
class HFBS {
 public:
  // This class holds the Cuda memory required, see hfbs.cu file.
  class HFBSImpl;

  // Creates a solver with max_image_dims as the maximum size  of the bilateral
  // space. Checks are done in Optimize() so that the solver does not use more
  // memory than specified by max_image_dims.
  HFBS(const ImageDim& max_image_dims, const int max_grid_x,
       const int max_grid_y, const int max_grid_l);

  // The destructor is required for proper deallocation of the memory held by
  // the unique_ptr.
  ~HFBS();

  // Sets the image size which is being processed.
  void SetImageDim(const ImageDim& image_dim);

  // This uploads the input color image (assumed 4 channel uchar image), target
  // and confidence. If color_space is not RGB, the input is converted to
  // color_space assuming the input is RGB. If confidence is not passed, its
  // assumed to be 1 for all pixels.
  void InitFrame(const COLOR_SPACE& color_space, void* color_image,
                 float* target, float* confidence = nullptr);

  // Sets the filter parameters for optimization.
  void SetFilterParams(const HFBSGridParams& grid_params);

  // Clears all the Cuda memory buffers to zero. This is useful for debugging
  // and for specifying an initial state for the solver.
  void ClearAll();

  // Runs the optimization using HFBSGridParams set using SetFilterParams().
  void Optimize();

  // Downloads the Cuda memory into CPU memory specified by image pointer. No
  // checks are done to ensure the same size.
  void Download(const ImageType& image_type, float* image) const;

 protected:
  // Initializes the color image by using the correct color space.
  void InitFrame(const COLOR_SPACE& color_space, void* color_image);

  // Holds the Cuda memory.
  std::unique_ptr<HFBSImpl> solver_impl_;

  // Dimensions of the image being processed.
  ImageDim image_dim_;

  // Parameters of the grid dimension and optimization.
  HFBSGridParams grid_params_;
};

}  // namespace domain_transform

#endif  // HFBS_H_
