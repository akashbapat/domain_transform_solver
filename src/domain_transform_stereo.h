#ifndef DOMAIN_TRANSFORM_STEREO_H_
#define DOMAIN_TRANSFORM_STEREO_H_

#include <memory>

#include "domain_transform_filter.h"
#include "domain_transform_solver.h"

namespace domain_transform {

class DomainTransformStereo : public DomainTransformSolver {
 public:
  /*
      domain_transform_stereo.SetImageDim(image_dim);

      domain_transform_stereo.InitFrame(domain_transform::COLOR_SPACE::RGB,
                                        solver_image.left_image.data,
                                        solver_image.target.data);

      domain_transform_stereo.InitRightFrame(solver_image.doffs,
                                             domain_transform::COLOR_SPACE::RGB,
                                             solver_image.right_image.data);

      domain_transform_stereo.ProcessRightFrame();
      constexpr int kLeftSideClearWidth = 1;
      domain_transform_stereo.ComputeConfidence(confidence_filter_params,
                                                kLeftSideClearWidth);

      domain_transform_stereo.ComputeColorSpaceDifferential(const
     DomainFilterParams& filter_params);
      domain_transform_stereo.IntegrateColorDifferentials();
      domain_transform_stereo.Optimize(optim_params);
  */

  explicit DomainTransformStereo(const ImageDim& max_image_dims);
  ~DomainTransformStereo();

  void InitRightFrame(const float center_offset, const COLOR_SPACE& color_space,
                      void* color_image);
  void ProcessRightFrame();

  void Optimize(const DomainOptimizeParams& optimize_params,
                const float overwrite_target_above_conf = -1) override;

  void SetImageDim(const ImageDim& image_dim) override;

 protected:
  std::unique_ptr<DomainTransformFilter> right_filter_;
  float center_offset_ = 0;
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_STEREO_H_
