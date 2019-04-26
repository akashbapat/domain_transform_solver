#ifndef DOMAIN_TRANSFORM_FILTER_H_
#define DOMAIN_TRANSFORM_FILTER_H_

#include <memory>

#include "cudaArray_fwd.h"

#include "domain_transform_common.h"
#include "domain_transform_filter_params.h"

namespace domain_transform {
// Forward declare the struct which holds the Cuda memory buffers.
class DomainTransformFilterStruct;
//
// This class implements the algorithm presented by Gastal and Oliveira, Domain
// Transform for Edge-Aware Image and Video Processing, with one change that the
// DT is defined with L2 norm as explained in Bapat and Frahm, The Domain
// Transform Solver.
class DomainTransformFilter {
 public:
  // max_image_dims defines the maximum size of the image that this class can
  // handle. max_image_dims is used to allocate GPU memory and is reused for
  // different images.
  explicit DomainTransformFilter(const ImageDim& max_image_dims);

  // The destructor is necessary for proper deallocation of the forward
  // declared DomainTransformFilterStruct.
  ~DomainTransformFilter();

  // Sets the dimension of the image to be filtered.
  virtual void SetImageDim(const ImageDim& image_dim);

  // Initializes the 4 channel uchar color image. If convert_to_color_space is
  // not RGB, the input color domain is assumed to be RGB and is used to convert
  // into a different colorspace.
  void InitFrame(const COLOR_SPACE& convert_to_color_space, void* color_image);

  // Filters input image and writes to output. No checks are done to ensure the
  // size of the memory at input and and is assumed to be whatever is set using
  // SetImageDim().
  void Filter(const DomainFilterParams& filter_params, const float* input,
              float* output, const int num_iter = 3);

  // Downloads the internal filter state, useful for debugging.  No checks are
  // done to ensure the size at pointer image is correct.
  virtual void Download(const ImageType& image_type, void* image) const;

  // Clears all buffers to zero.
  void ClearAll();

  // Do not use these if only using the filter.
  // Computes dHdx and dVdy.
  void ComputeColorSpaceDifferential(
      const DomainFilterParams& domain_filter_params);

  // Integrates dHdx and dVdy in parallel to obtain the domain transform values.
  void IntegrateColorDifferentials();

  DomainTransformFilterStruct* FilterStructPtr();

 protected:
  std::unique_ptr<DomainTransformFilterStruct> filter_struct_;
  ImageDim max_image_dims_;
  ImageDim image_dims_;

  // Useful function call to filter which directly uses the cuda memory buffers.
  void Filter(const DomainFilterParams& filter_params, const int num_iter,
              const cua::CudaArray2D<float>* input,
              cua::CudaArray2D<float>* output);
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_FILTER_H_
