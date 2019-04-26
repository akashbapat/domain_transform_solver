#include "pfm_image.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace domain_transform {
namespace {

// check whether machine is little endian.
PFMImage::Endian IsLittleEndian() {
  int intval = 1;
  unsigned char* uval = reinterpret_cast<unsigned char*>(&intval);
  return uval[0] == 1 ? PFMImage::Endian::LITTLE : PFMImage::Endian::BIG;
}

}  // namespace

PFMImage::PFMImage(const std::string& path_to_file) {
  // See \url{http://netpbm.sourceforge.net/doc/pfm.html} for description.
  std::ifstream pfm_image_file(path_to_file, std::ios::binary | std::ios::in);

  if (pfm_image_file.is_open()) {
    std::string header_str;
    pfm_image_file >> header_str;

    if (header_str.compare("Pf") == 0) {
      num_channels = 1;
    } else if (header_str.compare("PF") == 0) {
      num_channels = 3;
    }

    // This line contaons width then height.

    pfm_image_file >> width >> height;

    float endian_scale = -1;  // Negative meansin little endian, which is the
                              // default value for this struct.
    pfm_image_file >> endian_scale;

    // Remove the newline character.
    std::string line;
    getline(pfm_image_file, line);

    endianness = endian_scale < 0 ? Endian::LITTLE : Endian::BIG;
    const bool swap_bytes = IsLittleEndian() != Endian::LITTLE;

    scale = std::abs(endian_scale);

    // Now read the data from the file.
    data = std::unique_ptr<float[]>(new float[width * height * num_channels]);

    pfm_image_file.read(reinterpret_cast<char*>(data.get()),
                        4 * width * height * num_channels);

    if (swap_bytes) {
      std::cerr << "Swapping bytes to match endianness." << std::endl;
      for (int i = 0; i < width * height * num_channels; i++) {
        unsigned char tmp;
        unsigned char* ptr = reinterpret_cast<unsigned char*>(data.get() + i);
        tmp = ptr[0];
        ptr[0] = ptr[3];
        ptr[3] = tmp;
        tmp = ptr[1];
        ptr[1] = ptr[2];
        ptr[2] = tmp;
      }
    }
  }

  pfm_image_file.close();
}

}  // namespace domain_transform
