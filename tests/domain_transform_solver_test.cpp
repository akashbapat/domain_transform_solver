#include "domain_transform_solver.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "error_types.h"

namespace {

std::string domain_transform_solver_test_path_prefix = "";
bool domain_transform_stereo_update_out_files = false;

}  // namespace

namespace domain_transform {

class DomainTransformSolverTest : public ::testing::Test {
 protected:
  void SaveImage(const cv::Mat& image, const std::string& image_name,
                 const std::string& dir_name) {
    if (domain_transform_stereo_update_out_files) {
      cv::imwrite(domain_transform_solver_test_path_prefix + dir_name + "out_" +
                      image_name + ".png",
                  image);
    }
  }

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
    std::cerr << image_name << " min:" << min_val << " max:" << max_val
              << std::endl;
  }

  DomainFilterParams GetDomainFilterParams() {
    DomainFilterParams domain_filter_params;
    domain_filter_params.sigma_x = 8;
    domain_filter_params.sigma_y = 8;
    domain_filter_params.sigma_r = 0.2;

    return domain_filter_params;
  }

  DomainFilterParamsVec<1> GetConfParams() {
    DomainFilterParamsVec<1> domain_filter_params;
    domain_filter_params.sigma_x = 8;
    domain_filter_params.sigma_y = 8;
    domain_filter_params.sigma_r = 0.2;
    domain_filter_params.sigmas[0] = 16;
    return domain_filter_params;
  }

  DomainOptimizeParams GetOptimParams() {
    DomainOptimizeParams optim_params;
    optim_params.loss = RobustLoss::CHARBONNIER;
    optim_params.num_iterations = 100;
    optim_params.lambda = 0.99;
    optim_params.sigma_z = 5;
    optim_params.step_size = 0.99;
    optim_params.photo_consis_lambda = 0.02;
    return optim_params;
  }
};

TEST_F(DomainTransformSolverTest, ComputeCostVolumeWorks) {
  const std::string left_image_path =
      domain_transform_solver_test_path_prefix + "testdata/reference.png";
  const cv::Mat left_color_image = ReadRGBAImageFrom(left_image_path);

  ImageDim image_dims;
  image_dims.width = left_color_image.cols;
  image_dims.height = left_color_image.rows;

  DomainTransformSolver domain_transform_solver(image_dims);

  const cv::Mat confidence = ReadFloatImageFromUchar(
      domain_transform_solver_test_path_prefix + "testdata/confidence.png",
      1 / 255.0f);
  ImageStatistics(confidence, "confidence");
  constexpr float kDisparityScale = 1;
  const cv::Mat target = ReadFloatImageFromUchar(
      domain_transform_solver_test_path_prefix + "testdata/target.png",
      kDisparityScale);
  ImageStatistics(target, "target");

  domain_transform_solver.InitFrame(COLOR_SPACE::YCbCr, left_color_image.data,
                                    reinterpret_cast<float*>(target.data),
                                    reinterpret_cast<float*>(confidence.data));

  DomainFilterParamsVec<1> confidence_filter_params = GetConfParams();
  constexpr int kLeftSideClearWidth = 100;
  domain_transform_solver.ComputeConfidence(confidence_filter_params,
                                            kLeftSideClearWidth);

  domain_transform_solver.ComputeColorSpaceDifferential(
      GetDomainFilterParams());
  domain_transform_solver.IntegrateColorDifferentials();
  domain_transform_solver.Optimize(GetOptimParams());

  if (domain_transform_stereo_update_out_files) {
    cv::Mat confidence_image_computed = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    domain_transform_solver.Download(ImageType::CONFIDENCE,
                                     confidence_image_computed.data);

    ImageStatistics(confidence_image_computed, "confidence_image_computed");
    cv::imshow("confidence_image_computed", confidence_image_computed);

    cv::Mat color_image_test =
        cv::Mat(left_color_image.rows, left_color_image.cols, CV_8UC4,
                cv::Scalar(0, 0, 0, 0));

    domain_transform_solver.Download(ImageType::COLOR_IMAGE,
                                     color_image_test.data);

    ImageStatistics(color_image_test, "color_image_test");

    cv::Mat differential_image = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    domain_transform_solver.Download(ImageType::DIFFERENTIAL,
                                     differential_image.data);

    ImageStatistics(differential_image, "differential_image");
    //  cv::imshow("differential_image", differential_image);

    cv::Mat integral_image = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    domain_transform_solver.Download(ImageType::INTEGRAL, integral_image.data);

    ImageStatistics(integral_image, "integral_image");

    cv::Mat optim_disparity_image_with_nan = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    domain_transform_solver.Download(ImageType::OPTIMIZED_QUANTITY,
                                     optim_disparity_image_with_nan.data);

    ImageStatistics(optim_disparity_image_with_nan,
                    "optim_disparity_image_with_nan");

    cv::Mat mask = (optim_disparity_image_with_nan > 0) &
                   (optim_disparity_image_with_nan < 1000);
    cv::Mat optim_disparity_image = cv::Mat(
        left_color_image.rows, left_color_image.cols, CV_32FC1, cv::Scalar(0));

    optim_disparity_image_with_nan.copyTo(optim_disparity_image, mask);

    ImageStatistics(optim_disparity_image, "optim_disparity_image");

    SaveImage(optim_disparity_image / kDisparityScale, "solved", "testdata/");
    cv::Mat optim_disparity_image_norm;
    cv::normalize(optim_disparity_image, optim_disparity_image_norm);
    cv::imshow("solved", optim_disparity_image_norm * 255);

    cv::waitKey(0);
  }
}

}  // namespace domain_transform

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc == 2) {
    domain_transform_solver_test_path_prefix = argv[1];
    domain_transform_solver_test_path_prefix += "/";
  } else if (argc == 3) {
    domain_transform_solver_test_path_prefix = argv[1];
    domain_transform_solver_test_path_prefix += "/";
    domain_transform_stereo_update_out_files = argv[2];
  }
  return RUN_ALL_TESTS();
}
