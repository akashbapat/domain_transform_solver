#include "domain_transform_filter.h"

#include <gtest/gtest.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "error_types.h"

namespace {

std::string domain_transform_filter_test_path_prefix = "";
bool domain_transform_filter_test_visualize_results = false;

}  //  namespace

namespace domain_transform {

class DomainTransformFilterTest : public ::testing::Test {
 protected:
  cv::Mat ReadRGBAImageFrom(const std::string& full_name) {
    cv::Mat color_image = cv::imread(full_name, CV_LOAD_IMAGE_COLOR);
    cv::Mat rgba_image;
    cv::cvtColor(color_image, rgba_image, CV_BGR2RGBA);
    return rgba_image;
  }

  cv::Mat ReadFloatImageFromUchar(const std::string& full_name,
                                  const float scale) {
    cv::Mat gray_image = cv::imread(full_name, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat float_image;
    gray_image.convertTo(float_image, CV_32FC1, scale);
    return float_image;
  }

  void ImageStatistics(const cv::Mat& image, const std::string& image_name) {
    double max_val, min_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    std::cerr << image_name << " " << min_val << " " << max_val << std::endl;
  }
};

TEST_F(DomainTransformFilterTest, FilterWorks) {
  const std::string left_image_path =
      domain_transform_filter_test_path_prefix + "testdata/reference.png";
  const cv::Mat left_color_image = ReadRGBAImageFrom(left_image_path);

  ImageDim image_dims;
  image_dims.width = left_color_image.cols;
  image_dims.height = left_color_image.rows;

  DomainFilterParams domain_filter_params = {1.0f, 1.0f, 1.0f};
  domain_filter_params.sigma_x = 8;
  domain_filter_params.sigma_y = 8;
  domain_filter_params.sigma_r = 0.2;
  const int num_iterations = 1000;

  std::cerr << "Num iter : " << num_iterations
            << " sigma_x : " << domain_filter_params.sigma_x
            << " sigma_y : " << domain_filter_params.sigma_y
            << " sigma_r : " << domain_filter_params.sigma_r << std::endl;

  DomainTransformFilter domain_transform_filter(image_dims);

  constexpr float kDisparityScale = 1;
  const cv::Mat target = ReadFloatImageFromUchar(
      domain_transform_filter_test_path_prefix + "testdata/target.png",
      kDisparityScale);
  ImageStatistics(target, "target");

  domain_transform_filter.InitFrame(COLOR_SPACE::YCbCr, left_color_image.data);

  // Initialize the inputs.
  cv::Mat optim_disparity_image = cv::Mat(
      left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));
  domain_transform_filter.Filter(
      domain_filter_params, reinterpret_cast<float*>(target.data),
      reinterpret_cast<float*>(optim_disparity_image.data), num_iterations);

  // Visualize.
  if (domain_transform_filter_test_visualize_results) {
    cv::Mat differential_image = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    domain_transform_filter.Download(ImageType::DIFFERENTIAL,
                                     differential_image.data);

    ImageStatistics(differential_image, "differential_image");
    cv::imshow("differential_image", differential_image);

    cv::Mat integral_image = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    domain_transform_filter.Download(ImageType::INTEGRAL, integral_image.data);

    ImageStatistics(integral_image, "integral_image");
    cv::imshow("integral_image", integral_image / 255);

    cv::Mat optim_disparity_image_norm;
    cv::normalize(optim_disparity_image, optim_disparity_image_norm, 1, 0,
                  cv::NORM_L2, -1, optim_disparity_image > 0);

    cv::imshow("filtered_image", optim_disparity_image_norm * 255.0f);
    cv::Mat optim_dis_show_image = optim_disparity_image_norm * 255.0f;
    ImageStatistics(optim_disparity_image_norm, "optim_disparity_image_norm");
    ImageStatistics(optim_dis_show_image, "optim_dis_show_image");
    std::cerr << "Mean of the image is " << cv::mean(optim_dis_show_image)
              << std::endl;
    cv::waitKey(0);
  }
}

}  // namespace domain_transform

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc == 2) {
    domain_transform_filter_test_path_prefix = argv[1];
    domain_transform_filter_test_path_prefix += "/";
  } else if (argc == 3) {
    domain_transform_filter_test_path_prefix = argv[1];
    domain_transform_filter_test_path_prefix += "/";
    domain_transform_filter_test_visualize_results = argv[2];
  }
  return RUN_ALL_TESTS();
}
