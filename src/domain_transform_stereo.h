#ifndef DOMAIN_TRANSFORM_STEREO_H_
#define DOMAIN_TRANSFORM_STEREO_H_

#include <memory>

#include "domain_transform_filter.h"
#include "domain_transform_solver.h"

namespace domain_transform {
// This class solves the stereo problem using gradient descent in an edge aware
// sense. It solves the optimization function in Eq (7) in Bapat and Frahm, The
// domain Transform Solver. The typical usage of this calss is as follows:
//
// solver->SetImageDim(solver_options.image_dims);
//
// solver->ClearAll();
//
// solver->InitFrame(solver_options.color_space, rgba_left_image.data,
//                  reinterpret_cast<float*>(target_image.data),
//                  reinterpret_cast<float*>(confidence.data));
//
// solver->InitRightFrame(doff, solver_options.color_space,
// rgba_right_image.data);
//
// // Number of pixels from the left for which we want to clear the confidence.
// constexpr int kLeftSideClearWidth = 20;
// solver->ComputeConfidence(solver_options.conf_params, kLeftSideClearWidth);
//
// solver->ComputeColorSpaceDifferential(solver_options.filter_options);
// solver->IntegrateColorDifferentials();

// solver->ProcessRightFrame();
// solver->Optimize(solver_options.optim_options,
//                  solver_options.overwrite_target_above_conf);

class DomainTransformStereo : public DomainTransformSolver {
 public:
  explicit DomainTransformStereo(const ImageDim& max_image_dims);
  ~DomainTransformStereo();

  void InitRightFrame(const float center_offset, const COLOR_SPACE& color_space,
                      void* color_image);

  // Always call this after you call ComputeColorSpaceDifferential();
  void ProcessRightFrame();

  void Optimize(const DomainOptimizeParams& optimize_params,
                const float overwrite_target_above_conf = -1) override;

  void SetImageDim(const ImageDim& image_dim) override;

 protected:
  std::unique_ptr<DomainTransformFilter> right_filter_;
  // This is the offset in center pixel in left and right image. See doff in
  // Middlebury dataset for an example.
  float center_offset_ = 0;
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_STEREO_H_
