#ifndef DOMAIN_TRANSFORM_COMMON_H_
#define DOMAIN_TRANSFORM_COMMON_H_

namespace domain_transform {

// ImageDim is a handy way to specify the size of the image.
struct ImageDim {
  int width = 100;
  int height = 200;

  ImageDim Transpose() const {
    ImageDim dim;
    dim.width = height;
    dim.height = width;
    return dim;
  }
};

// This enum class defines type of image.
enum class ImageType {
  COLOR_IMAGE,
  DIFFERENTIAL,
  INTEGRAL,
  TARGET,
  CONFIDENCE,
  OPTIMIZED_QUANTITY,
  TEST
};

// This enum class defines color-space and useful during color-space
// transformation.
enum class COLOR_SPACE { RGB, YCbCr, YYY };

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_COMMON_H_
