#ifndef DOMAIN_TRANSFORM_IMAGE_IO_H_
#define DOMAIN_TRANSFORM_IMAGE_IO_H_

#include <opencv2/opencv.hpp>

#include <string>

namespace domain_transform {

void SaveImage(const cv::Mat& image, const std::string& image_name,
               const std::string& dir_name) {
  cv::imwrite(dir_name + "/" + image_name + ".png", image);
}

cv::Mat ReadRGBAImageFrom(const std::string& full_name) {
  cv::Mat color_image = cv::imread(full_name, CV_LOAD_IMAGE_COLOR);
  cv::Mat rgba_image = cv::Mat(color_image.rows, color_image.cols, CV_8UC4);
  int from_to[] = {0, 2, 1, 1, 2, 0, -1, 3};
  cv::mixChannels(&color_image, 1, &rgba_image, 1, from_to, 4);
  return rgba_image;
}

cv::Mat ConvertToRGBAImage(cv::Mat color_image) {
  cv::Mat rgba_image = cv::Mat(color_image.rows, color_image.cols, CV_8UC4);
  // cv::cvtColor(color_image, rgba_image, CV_BGR2RGBA);

  // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
  // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
  int from_to[] = {0, 2, 1, 1, 2, 0, -1, 3};
  cv::mixChannels(&color_image, 1, &rgba_image, 1, from_to, 4);

  return rgba_image;
}

cv::Mat ReadFloatImageFromUchar(const std::string& full_name,
                                const float scale) {
  cv::Mat gray_image = cv::imread(full_name, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat float_image;
  gray_image.convertTo(float_image, CV_32FC1, scale);
  return float_image;
}

cv::Mat ReadUcharImageFrom16bit(const std::string& full_name) {
  cv::Mat image = cv::imread(full_name, CV_LOAD_IMAGE_ANYDEPTH);
  cv::Mat uint8_image;
  image.convertTo(uint8_image, CV_8UC1, 1.0f / 256);
  return uint8_image;
}

cv::Mat ReadFloatImageFrom16bit(const std::string& full_name,
                                const float scale) {
  cv::Mat image = cv::imread(full_name, CV_LOAD_IMAGE_ANYDEPTH);
  cv::Mat float_image;
  image.convertTo(float_image, CV_32FC1, scale);
  return float_image;
}

void WriteFloatImageTo16bit(const std::string& full_name,
                            const cv::Mat& float_image, float* min_val,
                            float* max_val) {
  double min_val_local;
  double max_val_local;

  if (*min_val > *max_val) {
    cv::minMaxLoc(float_image, &min_val_local, &max_val_local);

    *min_val = min_val_local;
    *max_val = max_val_local;

  } else {
    min_val_local = *min_val;
    max_val_local = *max_val;
  }

  cv::Mat norm_image = cv::Mat(float_image.rows, float_image.cols, CV_32FC1);
  // Normalize image to 0 to 1.
  float_image.convertTo(norm_image, CV_32FC1,
                        1.0f / (max_val_local - min_val_local),
                        -min_val_local / (max_val_local - min_val_local));

  cv::Mat uint16_image = cv::Mat(float_image.rows, float_image.cols, CV_16UC1);

  norm_image.convertTo(uint16_image, CV_16UC1, 65535);

  cv::imwrite(full_name, uint16_image);
}

void ImageStatistics(const std::string& image_name, const cv::Mat& image) {
  double max_val, min_val;
  cv::minMaxLoc(image, &min_val, &max_val);
  std::cerr << image_name << " min:" << min_val << " max:" << max_val
            << std::endl;
}

}  //  namespace domain_transform

#endif  // DOMAIN_TRANSFORM_IMAGE_IO_H_
