#include <gflags/gflags.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "benchmark_util.h"
#include "depth_super_res_dataset.h"
#include "domain_transform_common.h"
#include "domain_transform_image_io.h"
#include "edge_aware_solver.h"

DEFINE_string(path_to_data_top_dir, "",
              "This is the path to the colorization data. The folder should "
              "hav the grayscale image, image with color stroes and gt.");

DEFINE_string(algorithm_name, "dts",
              "This is the algorithm name used to process the data. Options: "
              "dts | hfbs.");

namespace benchmark {

float PSNR(const cv::Mat& gt, const cv::Mat& test_image) {
  const double rmse = benchmark::RMSError(gt, test_image);

  const float psnr = 20 * std::log10(255) - 10 * std::log10(rmse);

  return psnr;
}

cv::Scalar PSNRCbCr(const cv::Mat& gt, const cv::Mat& test_image) {
  // Take the last two channels and calculate the per channel psnr.
  cv::Scalar psnr(0, 0, 0, 0);

  cv::Mat gt_ycbcr[3];
  cv::split(gt, gt_ycbcr);

  cv::Mat test_ycbcr[3];
  cv::split(test_image, test_ycbcr);

  psnr[1] = PSNR(gt_ycbcr[1], test_ycbcr[1]);

  psnr[2] = PSNR(gt_ycbcr[2], test_ycbcr[2]);

  return psnr;
}

cv::Scalar SSIM(const cv::Mat& test1, const cv::Mat& test2) {
  constexpr float kK1 = 0.01f;
  constexpr float kK2 = 0.03f;
  constexpr int kUint8Max = 255;

  const float C1 = (kK1 * kUint8Max) * (kK1 * kUint8Max);
  const float C2 = (kK2 * kUint8Max) * (kK2 * kUint8Max);
  constexpr int kRadius = 11;
  constexpr float kSigma = 1.5f;

  cv::Mat test1_fl, test2_fl;

  test1.convertTo(test1_fl, CV_32FC3);
  test2.convertTo(test2_fl, CV_32FC3);

  cv::Mat mu1, mu2;
  cv::GaussianBlur(test1_fl, mu1, cv::Size(kRadius, kRadius), kSigma, kSigma);

  cv::GaussianBlur(test2_fl, mu2, cv::Size(kRadius, kRadius), kSigma, kSigma);

  const cv::Mat mu1_sq = mu1.mul(mu1);
  const cv::Mat mu2_sq = mu2.mul(mu2);
  const cv::Mat mu1_mu2 = mu1.mul(mu2);

  cv::Mat sigma1_sq, sigma2_sq, sigma12;

  cv::GaussianBlur(test1_fl.mul(test1_fl), sigma1_sq,
                   cv::Size(kRadius, kRadius), kSigma, kSigma);

  sigma1_sq -= mu1_sq;

  cv::GaussianBlur(test2_fl.mul(test2_fl), sigma2_sq,
                   cv::Size(kRadius, kRadius), kSigma, kSigma);

  sigma2_sq -= mu2_sq;

  cv::GaussianBlur(test1_fl.mul(test2_fl), sigma12, cv::Size(kRadius, kRadius),
                   kSigma, kSigma);

  sigma12 -= mu1_mu2;

  const cv::Mat ssim_map =
      ((2 * mu1_mu2 + C1).mul(2 * sigma12 + C2)) /
      ((mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2));

  const cv::Scalar mssim = cv::mean(ssim_map);

  return mssim;
}

void BGR2YCbCr(const cv::Mat bgr_image, cv::Mat* y_image, cv::Mat* cb_image,
               cv::Mat* cr_image) {
  // Convert Tto floating point images.

  for (int row = 0; row < bgr_image.rows; row++) {
    const unsigned char* bgr_ptr = bgr_image.ptr<unsigned char>(row);

    for (int col = 0; col < bgr_image.cols; col++) {
      const unsigned char b = bgr_ptr[3 * col + 0];
      const unsigned char g = bgr_ptr[3 * col + 1];
      const unsigned char r = bgr_ptr[3 * col + 2];

      // See https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion for these
      // values.
      const float y = 0.299f * r + 0.587f * g + 0.114f * b;
      const float cb = 128 - 0.168736f * r - 0.331264f * g + 0.5f * b;
      const float cr = 128 + 0.5f * r - 0.418688f * g - 0.081312f * b;

      y_image->ptr<float>(row)[col] = y;
      cb_image->ptr<float>(row)[col] = cb;
      cr_image->ptr<float>(row)[col] = cr;
    }
  }
}

cv::Mat BGR2YCbCr(const cv::Mat bgr_image) {
  cv::Mat ycbcr = 0 * bgr_image;
  for (int row = 0; row < bgr_image.rows; row++) {
    const unsigned char* bgr_ptr = bgr_image.ptr<unsigned char>(row);

    for (int col = 0; col < bgr_image.cols; col++) {
      const unsigned char b = bgr_ptr[3 * col + 0];
      const unsigned char g = bgr_ptr[3 * col + 1];
      const unsigned char r = bgr_ptr[3 * col + 2];

      // See https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion for these
      // values.
      const float y = 0.299f * r + 0.587f * g + 0.114f * b;
      const float cb = 128 - 0.168736f * r - 0.331264f * g + 0.5f * b;
      const float cr = 128 + 0.5f * r - 0.418688f * g - 0.081312f * b;

      ycbcr.ptr<unsigned char>(row)[3 * col + 0] = y;
      ycbcr.ptr<unsigned char>(row)[3 * col + 1] = cb;
      ycbcr.ptr<unsigned char>(row)[3 * col + 2] = cr;
    }
  }

  return ycbcr;
}

cv::Mat YCbCrBGR(const cv::Mat y_image, const cv::Mat cb_image,
                 const cv::Mat cr_image) {
  // Convert to unit8.

  cv::Mat bgr_image = cv::Mat(y_image.rows, y_image.cols, CV_8UC3);

  for (int row = 0; row < bgr_image.rows; row++) {
    unsigned char* bgr_ptr = bgr_image.ptr<unsigned char>(row);
    for (int col = 0; col < bgr_image.cols; col++) {
      // See https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion for these
      // values.

      const float y = y_image.ptr<float>(row)[col];
      const float cb = cb_image.ptr<float>(row)[col];
      const float cr = cr_image.ptr<float>(row)[col];

      const float r = y + 1.402f * (cr - 128);
      const float g = y - 0.344136f * (cb - 128) - 0.714136 * (cr - 128);
      const float b = y + 1.772f * (cb - 128);

      bgr_ptr[3 * col + 0] = b > 255 ? 255 : b < 0 ? 0 : std::round(b);
      bgr_ptr[3 * col + 1] = g > 255 ? 255 : g < 0 ? 0 : std::round(g);
      bgr_ptr[3 * col + 2] = r > 255 ? 255 : r < 0 ? 0 : std::round(r);
    }
  }
  return bgr_image;
}

cv::Mat GetFirstChannelAsFloat(const cv::Mat& image) {
  cv::Mat y_image = cv::Mat(image.rows, image.cols, CV_32FC1);

  for (int row = 0; row < image.rows; row++) {
    const unsigned char* image_ptr = image.ptr<unsigned char>(row);
    float* y_image_ptr = y_image.ptr<float>(row);
    for (int col = 0; col < image.cols; col++) {
      y_image_ptr[col] = image_ptr[col * image.channels()];
    }
  }
  return y_image;
}

class ColorizationDataset {
 public:
  struct ColorizationData {
    cv::Mat yyya_image;
    cv::Mat cb_target;
    cv::Mat cb_conf;

    cv::Mat cr_target;
    cv::Mat cr_conf;

    cv::Mat ycbcr_gt;
    std::string image_name;
  };

  // There are 2 images in the dataset, change this if you add more.
  static constexpr int kNumImages = 2;

  explicit ColorizationDataset(const std::string& path_to_data_top_dir)
      : path_to_data_top_dir_(path_to_data_top_dir) {}

  // Defines the maximum size of the image in the entire dataset.
  domain_transform::ImageDim MaxImageDim() const {
    domain_transform::ImageDim max_image_dims;
    max_image_dims.width = 3280;
    max_image_dims.height = 3280;
    return max_image_dims;
  }

  // Reads the next image in the dataset. If there are no more images, this
  // starts again at the beginning.
  ColorizationData GetProblemImageData() {
    ColorizationData data;
    // Create yyya_image from the grayscale image.
    const std::string gray_image_path =
        path_to_data_top_dir_ + "/" + image_names_[im_idx_] + ".bmp";

    const cv::Mat gray_image_3_channel = cv::imread(gray_image_path);

    cv::Mat y_gray_image = GetFirstChannelAsFloat(gray_image_3_channel);

    cv::Mat gray_image;
    y_gray_image.convertTo(gray_image, CV_8UC1);

    int y_to_yyya[] = {0, 0, 0, 1, 0, 2, -1, 3};
    data.yyya_image = cv::Mat(y_gray_image.rows, y_gray_image.cols, CV_8UC4);
    cv::mixChannels(&gray_image, 1, &data.yyya_image, 1, y_to_yyya, 4);

    // Load the images with strokes and setup confidence and targets.
    const std::string marked_image_path =
        path_to_data_top_dir_ + "/" + image_names_[im_idx_] + "_strokes.bmp";

    const cv::Mat marked_image = cv::imread(marked_image_path);

    // Prepare data.
    cv::Mat y_image = cv::Mat(marked_image.rows, marked_image.cols, CV_32FC1);
    data.cb_target = cv::Mat(marked_image.rows, marked_image.cols, CV_32FC1);
    data.cr_target = cv::Mat(marked_image.rows, marked_image.cols, CV_32FC1);

    BGR2YCbCr(marked_image, &y_image, &data.cb_target, &data.cr_target);

    cv::Mat uint8_cb_image, uint8_cr_image;

    data.cb_target.convertTo(uint8_cb_image, CV_8UC1);
    data.cr_target.convertTo(uint8_cr_image, CV_8UC1);
    cv::Mat cb_confidence_image =
        ((uint8_cb_image > 129) | (uint8_cb_image < 127) |
         (uint8_cr_image > 129) | (uint8_cr_image < 127)) /
        255;
    cv::Mat cr_confidence_image =
        ((uint8_cr_image > 129) | (uint8_cr_image < 127) |
         (uint8_cb_image > 129) | (uint8_cb_image < 127)) /
        255;

    cb_confidence_image.convertTo(data.cb_conf, CV_32FC1);
    cr_confidence_image.convertTo(data.cr_conf, CV_32FC1);

    // Load the gt color image and convert to YCbCr.
    const std::string gt_full_path =
        path_to_data_top_dir_ + "/" + image_names_[im_idx_] + "_gt.bmp";

    data.ycbcr_gt = BGR2YCbCr(cv::imread(gt_full_path));

    data.image_name = image_names_[im_idx_];

    im_idx_++;

    if (im_idx_ >= image_names_.size()) {
      im_idx_ = 0;
    }

    return data;
  }

  // Assumes that the directory structure with the algorithm already exists.
  void SaveResultForAlogrithm(const std::string& algo_name,
                              const std::string& image_name,
                              const cv::Mat& result) {
    const std::string result_path =
        path_to_data_top_dir_ + "/" + algo_name + "_" + image_name + ".bmp";

    cv::imwrite(result_path, result);
  }

 private:
  const std::vector<std::string> image_names_ = {"IMG_3743", "IMG_3748"};

  std::string path_to_data_top_dir_;
  int im_idx_ = 0;
};

class DtsColorizationSolver : public domain_transform::EdgeAwareSolver<
                                  domain_transform::DomainTransformSolver,
                                  domain_transform::DtsOptions> {
 public:
  explicit DtsColorizationSolver(
      const domain_transform::ImageDim& max_image_dims) {
    solver_.reset(new domain_transform::DomainTransformSolver(max_image_dims));

    solver_options_.filter_options = GetDomainFilterParams();
    solver_options_.optim_options = GetOptimParams();
    solver_options_.conf_params = GetConfParams();
    // Dont change the input.
    solver_options_.color_space = domain_transform::COLOR_SPACE::RGB;
  }

  void PreProcess(cv::Mat* rgba_image, cv::Mat* target_image,
                  cv::Mat* confidence) override {}

  void PostProcessResult(cv::Mat* optimized_result) override {}

  domain_transform::DtsOptions ModifiedSolverOptions(
      const cv::Mat& rgba_image) {
    domain_transform::DtsOptions opt = options();
    opt.image_dims.height = rgba_image.rows;
    opt.image_dims.width = rgba_image.cols;
    opt.overwrite_target_above_conf = 0.85;
    return opt;
  }

 private:
  domain_transform::DomainFilterParamsVec<1> GetConfParams() {
    // These parameters do not matter since the confidence is loaded for depth
    // super resolution task.
    domain_transform::DomainFilterParamsVec<1> domain_filter_params;
    domain_filter_params.sigma_x = 8;
    domain_filter_params.sigma_y = 8;
    domain_filter_params.sigma_r = 0.2;
    domain_filter_params.sigmas[0] = 16;
    return domain_filter_params;
  }

  domain_transform::DomainOptimizeParams GetOptimParams() {
    domain_transform::DomainOptimizeParams optim_params;
    optim_params.loss = domain_transform::RobustLoss::CHARBONNIER;
    optim_params.num_iterations = 10;
    optim_params.lambda = 0.99;
    optim_params.step_size = 0.99;
    optim_params.sigma_z = 30;
    optim_params.photo_consis_lambda = 0.0;
    optim_params.DT_lambda = 0.0;
    return optim_params;
  }

  domain_transform::DomainFilterParams GetDomainFilterParams() {
    domain_transform::DomainFilterParams domain_filter_params;
    domain_filter_params.sigma_x = 64;
    domain_filter_params.sigma_y = 64;
    domain_filter_params.sigma_r = 0.25;

    return domain_filter_params;
  }
};

class HfbsColorizationSolver
    : public domain_transform::EdgeAwareSolver<domain_transform::HFBS,
                                               domain_transform::HfbsOptions> {
 public:
  explicit HfbsColorizationSolver(
      const domain_transform::ImageDim& max_image_dims) {
    solver_.reset(new domain_transform::HFBS(
        max_image_dims, std::ceil(3280.0 / 8), std::ceil(3280.0 / 8), 64));
    // Change the max grid size above according to the grid sigmas if out of
    // memory.
    solver_options_.grid_params.sigma_x = 8;
    solver_options_.grid_params.sigma_y = 8;
    solver_options_.grid_params.sigma_l = 4;
    solver_options_.grid_params.optim_iter_max = 140;
    // solver_options_.grid_params.optim_iter_max = 27; // same time budget.
    solver_options_.grid_params.bistoch_iter_max = 20;

    solver_options_.grid_params.lambda = 8.0f;

    // Dont change the input.
    solver_options_.color_space = domain_transform::COLOR_SPACE::RGB;
  }

  void PreProcess(cv::Mat* rgba_image, cv::Mat* target_image,
                  cv::Mat* confidence) override {
    *target_image = *target_image - 128;
    *confidence = (*confidence) * 110000000164;
  }

  void PostProcessResult(cv::Mat* optimized_result) override {
    *optimized_result = *optimized_result + 128;
  }

  domain_transform::HfbsOptions ModifiedSolverOptions(
      const cv::Mat& rgba_image) {
    domain_transform::HfbsOptions opt = options();
    opt.image_dims.height = rgba_image.rows;
    opt.image_dims.width = rgba_image.cols;

    return opt;
  }
};

template <typename EdgeAwareSolverType>
void BenchmarkColorization(const std::string& path_to_data_top_dir,
                           const std::string& algorithm_name) {
  benchmark::BenchmarkResult<benchmark::ColorizationDataset::kNumImages> bres;

  constexpr int kNumTrials = 1;
  for (int trial_idx = 0; trial_idx < kNumTrials; trial_idx++) {
    benchmark::ColorizationDataset dataset(path_to_data_top_dir);
    EdgeAwareSolverType problem_solver(dataset.MaxImageDim());

    for (int im_idx = 0; im_idx < benchmark::ColorizationDataset::kNumImages;
         im_idx++) {
      ColorizationDataset::ColorizationData image_data =
          dataset.GetProblemImageData();
      // Solve for Cb channel.
      cv::Mat cb_estimate =
          cv::Mat(image_data.yyya_image.rows, image_data.yyya_image.cols,
                  CV_32FC1, cv::Scalar(0));

      problem_solver.PreProcess(&image_data.yyya_image, &image_data.cb_target,
                                &image_data.cb_conf);
      const float local_time_cb_in_ms = domain_transform::EdgeAwareOptimize(
          image_data.yyya_image, image_data.cb_target, image_data.cb_conf,
          problem_solver.ModifiedSolverOptions(image_data.yyya_image),
          &problem_solver.solver(), &cb_estimate);
      problem_solver.PostProcessResult(&cb_estimate);
      // Solve for Cr channel.
      cv::Mat cr_estimate =
          cv::Mat(image_data.yyya_image.rows, image_data.yyya_image.cols,
                  CV_32FC1, cv::Scalar(0));

      problem_solver.PreProcess(&image_data.yyya_image, &image_data.cr_target,
                                &image_data.cr_conf);

      const float local_time_cr_in_ms = domain_transform::EdgeAwareOptimize(
          image_data.yyya_image, image_data.cr_target, image_data.cr_conf,
          problem_solver.ModifiedSolverOptions(image_data.yyya_image),
          &problem_solver.solver(), &cr_estimate);

      problem_solver.PostProcessResult(&cr_estimate);

      const float local_time_in_ms = local_time_cb_in_ms + local_time_cr_in_ms;

      bres.AddTrial(im_idx, 0, local_time_in_ms,
                    image_data.yyya_image.rows * image_data.yyya_image.cols);

      // Save the result.
      cv::Mat bgr_estimate =
          YCbCrBGR(GetFirstChannelAsFloat(image_data.yyya_image), cb_estimate,
                   cr_estimate);
      dataset.SaveResultForAlogrithm(algorithm_name, image_data.image_name,
                                     bgr_estimate);

      std::cout << image_data.image_name << ", SSIM: "
                << SSIM(image_data.ycbcr_gt, BGR2YCbCr(bgr_estimate))
                << ", PSNR: "
                << PSNRCbCr(image_data.ycbcr_gt, BGR2YCbCr(bgr_estimate))
                << std::endl;
    }
  }

  bres.Normalize(kNumTrials);

  std::cerr << "RMSE " << bres.rmse() << ", time " << bres.time()
            << " ms, num_pixels " << bres.num_pixels() << std::endl;
}

}  // namespace benchmark

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (std::string(FLAGS_path_to_data_top_dir).empty() ||
      std::string(FLAGS_algorithm_name).empty()) {
    std::cerr << "Invalid command line inputs." << std::endl;
    return -1;
  }

  const std::string algo_name = FLAGS_algorithm_name;

  if (algo_name == "dts") {
    std::cout << "Using " << algo_name << " algorithm." << std::endl;
    benchmark::BenchmarkColorization<benchmark::DtsColorizationSolver>(
        FLAGS_path_to_data_top_dir, FLAGS_algorithm_name);
  } else if (algo_name == "hfbs") {
    std::cout << "Using " << algo_name << " algorithm." << std::endl;
    benchmark::BenchmarkColorization<benchmark::HfbsColorizationSolver>(
        FLAGS_path_to_data_top_dir, FLAGS_algorithm_name);
  } else {
    std::cerr << "Unsupported algorithm." << std::endl;
    return -1;
  }
}
