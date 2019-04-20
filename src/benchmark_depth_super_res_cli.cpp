#include <gflags/gflags.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "benchmark_util.h"
#include "depth_super_res_dataset.h"
#include "domain_transform_common.h"
#include "domain_transform_image_io.h"
#include "edge_aware_solver.h"

DEFINE_string(
    path_to_data_top_dir, "",
    "This is the path to the data as provided by Jon Barron(from his site).");

DEFINE_string(algorithm_name, "dts",
              "This is the algorithm name used to process the data. Options: "
              "dts | hfbs.");

DEFINE_string(algorithm_input_name, "bicubic",
              "This is the algorithm name used as the input.");

namespace benchmark {

class DtsSuperResSolver : public domain_transform::EdgeAwareSolver<
                              domain_transform::DomainTransformSolver,
                              domain_transform::DtsOptions> {
 public:
  explicit DtsSuperResSolver(const domain_transform::ImageDim& max_image_dims) {
    solver_.reset(new domain_transform::DomainTransformSolver(max_image_dims));

    solver_options_.filter_options = GetDomainFilterParams();
    solver_options_.optim_options = GetOptimParams();
    solver_options_.conf_params = GetConfParams();
    solver_options_.color_space = domain_transform::COLOR_SPACE::YCbCr;
  }

  void PreProcess(cv::Mat* rgba_image, cv::Mat* target_image,
                  cv::Mat* confidence) override {}

  void PostProcessResult(cv::Mat* optimized_result) override {
    optimized_result->convertTo((*optimized_result), CV_8UC1, 256.0f - 1);
  }

  domain_transform::DtsOptions ModifiedSolverOptions(
      const int upsample, const cv::Mat& rgba_image) {
    domain_transform::DtsOptions opt = options();
    opt.filter_options.sigma_x = 8 * upsample;
    opt.filter_options.sigma_y = 8 * upsample;
    opt.optim_options.sigma_z = 8 * upsample / (1 + 0.2 * 4);
    opt.image_dims.height = rgba_image.rows;
    opt.image_dims.width = rgba_image.cols;
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
    optim_params.loss = domain_transform::RobustLoss::L2;
    optim_params.num_iterations = 10;
    optim_params.lambda = 0.99;
    optim_params.step_size = 0.99;
    optim_params.photo_consis_lambda = 0.02;
    return optim_params;
  }

  domain_transform::DomainFilterParams GetDomainFilterParams() {
    domain_transform::DomainFilterParams domain_filter_params;
    domain_filter_params.sigma_x = 8;
    domain_filter_params.sigma_y = 8;
    domain_filter_params.sigma_r = 0.1;

    return domain_filter_params;
  }
};

class HfbsSuperResSolver
    : public domain_transform::EdgeAwareSolver<domain_transform::HFBS,
                                               domain_transform::HfbsOptions> {
 public:
  explicit HfbsSuperResSolver(
      const domain_transform::ImageDim& max_image_dims) {
    solver_.reset(new domain_transform::HFBS(
        max_image_dims, max_image_dims.width, max_image_dims.height, 64));

    solver_options_.grid_params.sigma_x = 8;
    solver_options_.grid_params.sigma_y = 8;
    solver_options_.grid_params.sigma_l = 16;

    solver_options_.color_space = domain_transform::COLOR_SPACE::YCbCr;
  }

  void PreProcess(cv::Mat* rgba_image, cv::Mat* target_image,
                  cv::Mat* confidence) override {
    (*target_image) = ((*target_image) - 0.5) * 128;
  }

  void PostProcessResult(cv::Mat* optimized_result) override {
    cv::Mat optim_disparity_image = (*optimized_result) / 128.0f + 0.5;

    optim_disparity_image.convertTo((*optimized_result), CV_8UC1, 256.0f - 1);
  }

  domain_transform::HfbsOptions ModifiedSolverOptions(
      const int upsample, const cv::Mat& rgba_image) {
    std::cerr << upsample << std::endl;
    domain_transform::HfbsOptions opt = options();
    opt.grid_params.sigma_x = 8 * upsample;
    opt.grid_params.sigma_y = 8 * upsample;

    opt.image_dims.height = rgba_image.rows;
    opt.image_dims.width = rgba_image.cols;

    return opt;
  }
};

template <typename EdgeAwareSolverType>
void BenchmarkDepthSuperResolution(const std::string& path_to_data_top_dir,
                                   const std::string& algorithm_name,
                                   const std::string& algorithm_input_name) {
  benchmark::BenchmarkResult<benchmark::DepthSuperResData::kNumImages> bres;

  constexpr int kNumTrials = 1;
  for (int trial_idx = 0; trial_idx < kNumTrials; trial_idx++) {
    benchmark::DepthSuperResData dataset(path_to_data_top_dir,
                                         algorithm_input_name);
    EdgeAwareSolverType problem_solver(dataset.MaxImageDim());

    for (int im_idx = 0; im_idx < benchmark::DepthSuperResData::kNumImages;
         im_idx++) {
      int upsample_level = 1;
      domain_transform::ProblemImageData image_data =
          dataset.GetProblemImageData(&upsample_level);

      problem_solver.PreProcess(&image_data.rgba_image, &image_data.target,
                                &image_data.confidence);

      cv::Mat optim_disparity_image =
          cv::Mat(image_data.rgba_image.rows, image_data.rgba_image.cols,
                  CV_32FC1, cv::Scalar(0));

      const float local_time_in_ms = domain_transform::EdgeAwareOptimize(
          image_data.rgba_image, image_data.target, image_data.confidence,
          problem_solver.ModifiedSolverOptions(upsample_level,
                                               image_data.rgba_image),
          &problem_solver.solver(), &optim_disparity_image);

      problem_solver.PostProcessResult(&optim_disparity_image);

      const double rms_error =
          benchmark::RMSError(image_data.gt, optim_disparity_image);

      bres.AddTrial(im_idx, rms_error, local_time_in_ms,
                    optim_disparity_image.rows * optim_disparity_image.cols);

      dataset.SaveResultForAlogrithm(algorithm_name, optim_disparity_image);
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
      std::string(FLAGS_algorithm_name).empty() ||
      std::string(FLAGS_algorithm_input_name).empty()) {
    std::cerr << "Invalid command line inputs." << std::endl;
    return -1;
  }

  const std::string algo_name = FLAGS_algorithm_name;

  if (algo_name == "dts") {
    std::cout << "Using " << algo_name << " algorithm." << std::endl;
    benchmark::BenchmarkDepthSuperResolution<benchmark::DtsSuperResSolver>(
        FLAGS_path_to_data_top_dir, FLAGS_algorithm_name,
        FLAGS_algorithm_input_name);
  } else if (algo_name == "hfbs") {
    std::cout << "Using " << algo_name << " algorithm." << std::endl;
    benchmark::BenchmarkDepthSuperResolution<benchmark::HfbsSuperResSolver>(
        FLAGS_path_to_data_top_dir, FLAGS_algorithm_name,
        FLAGS_algorithm_input_name);
  } else {
    std::cerr << "Unsupported algorithm." << std::endl;
    return -1;
  }
}
