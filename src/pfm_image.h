#ifndef PFM_IMAGE_H_
#define PFM_IMAGE_H_

#include <memory>
#include <string>

namespace domain_transform {

struct PFMImage {
  enum class Endian { LITTLE, BIG };
  explicit PFMImage(const std::string& path_to_file);

  int width = 0;
  int height = 0;
  int num_channels = 0;
  Endian endianness = Endian::LITTLE;
  float scale = 1;
  std::unique_ptr<float[]> data;
};

}  // namespace domain_transform

#endif  // PFM_IMAGE_H_
