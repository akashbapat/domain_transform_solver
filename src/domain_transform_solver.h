#ifndef DOMAIN_TRANSFORM_SOLVER_H_
#define DOMAIN_TRANSFORM_SOLVER_H_

#include <memory>

#include "domain_transform_common.h"
#include "domain_transform_filter.h"
#include "domain_transform_filter_params.h"
#include "domain_transform_optimize.h"

namespace domain_transform {
// Forward declaration for struct which holds all the cuda memory buffers
// rquired for the solver in addition to the buffers in
// DomainTransformFilterStruct.
class DomainTransformSolverStruct;

// This class implements a bare-bones algorithm of the method presented in Bapat
// and Frahm, The Domain Transform Solver paper.
// A typical useage is as follows:
//
//  DomainTransformSolver domain_transform_solver(image_dims);
//  // Load the images.
//  cv::Mat confidence, target, color_image;
//
//  domain_transform_solver.InitFrame(COLOR_SPACE::YCbCr, left_color_image.data,
//                                    reinterpret_cast<float*>(target.data),
//                                    reinterpret_cast<float*>(confidence.data));
//
//  domain_transform_solver.ComputeColorSpaceDifferential(GetDomainFilterParams());
//  domain_transform_solver.IntegrateColorDifferentials();
//  domain_transform_solver.Optimize(GetOptimParams())

class DomainTransformSolver : public DomainTransformFilter {
 public:
  // max_image_dims defines the maximum size of the image that this class can
  // handle. max_image_dims is used to allocate GPU memory and is reused for
  // different images.
  explicit DomainTransformSolver(const ImageDim& max_image_dims);

  // The destructor is necessary for proper deallocation of the forward
  // declared DomainTransformSolverStruct.
  ~DomainTransformSolver();

  // Initializes the 4 channel uchar color image. If color_space is not RGB, the
  // input color domain is assumed to be RGB and is used to convert into a
  // different colorspace. The target is the term 't' in Eqn 2 from the paper.
  // If confidence values are available for the target, the solver uses it as
  // input. Otherwise, the solver automatically calculates an edge-aware
  // confidence measure using the domain transform as expectation as suggested
  // by Barron and Poole, The fast bilateral solver (supplementary) Eqn 21-23.
  void InitFrame(const COLOR_SPACE& color_space, void* color_image,
                 float* target, float* confidence = nullptr);

  // Initializes the computation of the domain transform by computing the
  // derivate of DT.
  void ComputeColorSpaceDifferential(const DomainFilterParams& filter_params);

  // Integrates dDT to obtain DT.
  void IntegrateColorDifferentials();

  // Runs the solver with gradient descent scheme. IF
  // overwrite_target_above_conf is between (0,1), the target pixels with
  // confidence more than overwrite_target_above_conf is overwritten to the
  // solution  in every iteration of gradient descent.
  virtual void Optimize(const DomainOptimizeParams& optimize_params,
                        const float overwrite_target_above_conf = -1);

  void Download(const ImageType& image_type, void* image) const override;

  // Computes the confidence using DT as an expectation of edge-aware mean.
  // left_side_clear_width will set confidence to 0 from left, useful in stereo
  // problem.
  void ComputeConfidence(const DomainFilterParamsVec<1>& conf_params,
                         const int left_side_clear_width);

  // Sets all the cuda memory buffers to 0.
  void ClearAll();

 protected:
  std::unique_ptr<DomainTransformSolverStruct> solver_struct_;
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_SOLVER_H_
