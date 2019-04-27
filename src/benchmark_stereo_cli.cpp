#include <gflags/gflags.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "benchmark_util.h"
#include "depth_super_res_dataset.h"
#include "domain_transform_common.h"
#include "domain_transform_image_io.h"
#include "domain_transform_stereo.h"
#include "edge_aware_solver.h"
#include "pfm_image.h"

DEFINE_string(path_to_data_top_dir, "", "This is the path to the stereo data.");

namespace benchmark {

std::vector<std::string> split_str(const std::string& _str,
                                   const std::string& delim) {
  // Copy the input string so that we dont destroy it.
  std::string str = _str;
  char* curr;
  std::vector<std::string> tokens;
  curr = strtok(const_cast<char*>(str.c_str()), delim.c_str());
  while (curr != nullptr) {
    tokens.push_back(curr);
    curr = std::strtok(nullptr, delim.c_str());
  }
  return tokens;
}

class StereoDataset {
 public:
  struct StereoData {
    cv::Mat rgba_left_image;
    cv::Mat rgba_right_image;
    cv::Mat disparity_target;
    // This is empty as conf is not provided.
    cv::Mat conf;

    int doff;
    cv::Mat disparity_gt;
    std::string image_name;
  };

  static constexpr int kNumImages = 1;

  explicit StereoDataset(const std::string& path_to_data_top_dir)
      : path_to_data_top_dir_(path_to_data_top_dir) {}

  // Defines the maximum size of the image in the entire dataset.
  domain_transform::ImageDim MaxImageDim() const {
    domain_transform::ImageDim max_image_dims;
    max_image_dims.width = 768;
    max_image_dims.height = 768;
    return max_image_dims;
  }

  // Reads the next image in the dataset. If there are no more images, this
  // starts again at the beginning.
  StereoData GetProblemImageData() {
    StereoData data;
    data.rgba_left_image = domain_transform::ReadRGBAImageFrom(
        path_to_data_top_dir_ + "/" + folder_names_[im_idx_] + "/" +
        image_names_left_[im_idx_] + ".png");

    data.rgba_right_image = domain_transform::ReadRGBAImageFrom(
        path_to_data_top_dir_ + "/" + folder_names_[im_idx_] + "/" +
        image_names_right_[im_idx_] + ".png");

    data.disparity_target =
        LoadTargetImage(path_to_data_top_dir_ + "/" + folder_names_[im_idx_] +
                            "/" + image_names_left_[im_idx_] + ".png_disp.bin",
                        data.rgba_left_image);
    // Read the offset in center pixel if present.
    data.doff = ReadDoffFromFile(path_to_data_top_dir_ + "/" +
                                 folder_names_[im_idx_] + "/" + kCalibFilename);

    if (!ReadDisparityPfm(path_to_data_top_dir_ + "/" + folder_names_[im_idx_] +
                              "/" + gt_names_[im_idx_] + ".pfm",
                          data.rgba_left_image, &data.disparity_gt)) {
      std::cerr << "Could not read the groundtruth disparoty pfm file."
                << std::endl;
    }

    data.image_name = folder_names_[im_idx_];

    im_idx_++;

    if (im_idx_ >= image_names_left_.size()) {
      im_idx_ = 0;
    }

    return data;
  }

  // Assumes that the directory structure with the algorithm already exists.
  void SaveResultForAlogrithm(const std::string& algo_name,
                              const std::string& image_name,
                              const cv::Mat& result) {
    const std::string result_path =
        path_to_data_top_dir_ + "/" + algo_name + "_" + image_name + ".png";

    cv::imwrite(result_path, result);
  }

 private:
  const std::vector<std::string> image_names_left_ = {"im0"};
  const std::vector<std::string> image_names_right_ = {"im1"};
  const std::vector<std::string> gt_names_ = {"disp0GT"};
  const std::vector<std::string> folder_names_ = {"Art"};
  const std::string kCalibFilename = "calib.txt";
  std::string path_to_data_top_dir_;
  int im_idx_ = 0;

  int ReadDoffFromFile(const std::string& file_path) {
    std::ifstream calib_file(file_path);
    int doff = 0;
    if (calib_file.is_open()) {
      // 3rd line is doff.
      std::string waste;
      std::getline(calib_file, waste);
      std::getline(calib_file, waste);
      calib_file >> waste;
      calib_file.close();

      const std::vector<std::string> tokens = split_str(waste, "=");
      if (tokens.size() == 2) {
        doff = std::stoi(tokens[1]);
      }

    } else {
      std::cerr << "Could not open calibration file at " << file_path
                << std::endl;
    }
    return doff;
  }

  bool ReadDisparityPfm(const std::string& fname,
                        const cv::Mat& left_rgba_image, cv::Mat* disparity) {
    domain_transform::PFMImage pfm_image(fname);
    if (left_rgba_image.cols != pfm_image.width ||
        left_rgba_image.rows != pfm_image.height ||
        pfm_image.num_channels != 1) {
      return false;
    }
    (*disparity) =
        cv::Mat(pfm_image.height, pfm_image.width, CV_32FC1, cv::Scalar(0));

    // Convert the raw diparity map to depth in mm.
    for (int row = 0; row < (*disparity).rows; row++) {
      float* row_ptr = (*disparity).ptr<float>((*disparity).rows - row - 1);
      for (int col = 0; col < (*disparity).cols; col++) {
        // Middlebury saves the disparity flipped.
        row_ptr[col] = pfm_image.data[row * (*disparity).cols + col];
        if (std::isinf(row_ptr[col])) {
          row_ptr[col] = 0;
        }
      }
    }
    return true;
  }

  cv::Mat LoadTargetImage(const std::string& disp_bin_target_fname,
                          const cv::Mat& rgba_left_image) {
    // Load the target image.
    cv::Mat target = cv::Mat(rgba_left_image.rows, rgba_left_image.cols,
                             CV_32FC1, cv::Scalar(0));

    const int num_pixels = rgba_left_image.rows * rgba_left_image.cols;

    std::ifstream disp_binary((disp_bin_target_fname),
                              std::ios::in | std::ios::binary);

    if (disp_binary.is_open()) {
      // Read the disparity values to a block of memory.
      std::unique_ptr<float[]> data =
          std::unique_ptr<float[]>(new float[num_pixels]);

      disp_binary.read(reinterpret_cast<char*>(data.get()),
                       num_pixels * sizeof(float));

      disp_binary.close();
      // Write to target.

      for (int row = 0; row < target.rows; row++) {
        float* row_ptr = target.ptr<float>(row);
        for (int col = 0; col < target.cols; col++) {
          row_ptr[col] = data[row * target.cols + col];
        }
      }

    } else {
      std::cerr << "Couldnt read the binary initialization at "
                << disp_bin_target_fname << std::endl;
    }
    return target;
  }
};

class DtsStereoSolver : public domain_transform::EdgeAwareSolver<
                            domain_transform::DomainTransformStereo,
                            domain_transform::DtsOptions> {
 public:
  explicit DtsStereoSolver(const domain_transform::ImageDim& max_image_dims) {
    solver_.reset(new domain_transform::DomainTransformStereo(max_image_dims));

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
    return opt;
  }

 private:
  domain_transform::DomainFilterParamsVec<1> GetConfParams() {
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
    optim_params.num_iterations = 1;
    optim_params.lambda = 0.99;
    optim_params.step_size = 0.99;
    optim_params.sigma_z = 30;
    optim_params.photo_consis_lambda = 0.01;
    optim_params.DT_lambda = 0.0;
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

template <typename SolverOptions, typename SolverType>
float EdgeAwareStereo(const cv::Mat& rgba_left_image,
                      const cv::Mat& target_image, const cv::Mat& confidence,
                      const cv::Mat& rgba_right_image,
                      const SolverOptions& solver_options, const int doff,
                      SolverType* solver, cv::Mat* optim_image) {
  solver->SetImageDim(solver_options.image_dims);

  solver->ClearAll();

  solver->InitFrame(solver_options.color_space, rgba_left_image.data,
                    reinterpret_cast<float*>(target_image.data),
                    reinterpret_cast<float*>(confidence.data));

  solver->InitRightFrame(doff, solver_options.color_space,
                         rgba_right_image.data);

  // Start Logging time.
  cudaEvent_t start, stop;
  GPU_CHECK(cudaEventCreate(&start));
  GPU_CHECK(cudaEventCreate(&stop));
  GPU_CHECK(cudaEventRecord(start, 0));

  // Number of pixels from the left for which we want to clear the confidence.
  constexpr int kLeftSideClearWidth = 20;
  solver->ComputeConfidence(solver_options.conf_params, kLeftSideClearWidth);

  solver->ComputeColorSpaceDifferential(solver_options.filter_options);
  solver->IntegrateColorDifferentials();

  solver->ProcessRightFrame();
  solver->Optimize(solver_options.optim_options,
                   solver_options.overwrite_target_above_conf);

  // Stop logging time.
  GPU_CHECK(cudaEventRecord(stop, 0));
  GPU_CHECK(cudaEventSynchronize(stop));
  float local_time_in_ms = 0;
  GPU_CHECK(cudaEventElapsedTime(&local_time_in_ms, start, stop));

  solver->Download(domain_transform::ImageType::OPTIMIZED_QUANTITY,
                   reinterpret_cast<float*>((*optim_image).data));

  return local_time_in_ms;
}

template <typename EdgeAwareSolverType>
void BenchmarkStereo(const std::string& path_to_data_top_dir,
                     const std::string& algorithm_name) {
  benchmark::BenchmarkResult<benchmark::StereoDataset::kNumImages> bres;

  constexpr int kNumTrials = 1;
  for (int trial_idx = 0; trial_idx < kNumTrials; trial_idx++) {
    benchmark::StereoDataset dataset(path_to_data_top_dir);
    EdgeAwareSolverType problem_solver(dataset.MaxImageDim());

    for (int im_idx = 0; im_idx < benchmark::StereoDataset::kNumImages;
         im_idx++) {
      StereoDataset::StereoData image_data = dataset.GetProblemImageData();

      cv::Mat disparity_estimate =
          cv::Mat(image_data.rgba_left_image.rows,
                  image_data.rgba_left_image.cols, CV_32FC1, cv::Scalar(0));

      problem_solver.PreProcess(&image_data.rgba_left_image,
                                &image_data.disparity_target, &image_data.conf);

      const float local_time_in_ms = EdgeAwareStereo(
          image_data.rgba_left_image, image_data.disparity_target,
          image_data.conf, image_data.rgba_right_image,
          problem_solver.ModifiedSolverOptions(image_data.rgba_left_image),
          image_data.doff, &problem_solver.solver(), &disparity_estimate);

      problem_solver.PostProcessResult(&disparity_estimate);

      domain_transform::ImageStatistics("disparity_estimate",
                                        disparity_estimate);
      domain_transform::ImageStatistics("image_data.disparity_gt",
                                        image_data.disparity_gt);
      bres.AddTrial(
          im_idx,
          benchmark::RMSError(image_data.disparity_gt, disparity_estimate),
          local_time_in_ms,
          image_data.rgba_left_image.rows * image_data.rgba_left_image.cols);

      // Save the result.
      dataset.SaveResultForAlogrithm(algorithm_name, image_data.image_name,
                                     disparity_estimate);
    }
  }

  bres.Normalize(kNumTrials);

  std::cerr << "RMSE " << bres.rmse() << ", time " << bres.time()
            << " ms, num_pixels " << bres.num_pixels() << std::endl;
}

}  // namespace benchmark

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (std::string(FLAGS_path_to_data_top_dir).empty()) {
    std::cerr << "Invalid command line inputs." << std::endl;
    return -1;
  }
  std::cout << "Using DTS algorithm." << std::endl;
  benchmark::BenchmarkStereo<benchmark::DtsStereoSolver>(
      FLAGS_path_to_data_top_dir, "dts");
}
