#ifndef BENCHMARK_UTIL_H_
#define BENCHMARK_UTIL_H_

#include <opencv2/opencv.hpp>
#include "domain_transform_filter_params.h"
#include "domain_transform_optimize.h"

namespace benchmark {

// BenchmarkResult provides useful utility functions to compare the accuracy and
// runtime for different methods over multiple trials. N is the number of images
// in the dataset. Use the function Normalize() at the end to compute the
// average time and accuracy.
template <int N>
class BenchmarkResult {
 public:
  static constexpr int kNumImages = N;

  void AddTrial(const int id, const double& rmse, const double time,
                const int num_pixels) {
    per_image_rmse_[id] += std::log(rmse);

    per_image_time_[id] += time;

    per_image_pixels_[id] += num_pixels;
  }

  // Normalizes the time, accuracy and the number of pixels over num_trials.
  void Normalize(const int num_trials) {
    rmse_ = 0;
    time_ = 0;
    num_pixels_ = 0;
    for (int i = 0; i < kNumImages; i++) {
      rmse_ += per_image_rmse_[i];
      time_ += per_image_time_[i];
      num_pixels_ += per_image_pixels_[i];
    }

    rmse_ /= (kNumImages * num_trials * 1.0f);
    time_ /= (kNumImages * num_trials * 1.0f);

    num_pixels_ /= (kNumImages * num_trials * 1.0f);

    rmse_ = std::exp(rmse_);
  }

  double time() const { return time_; }

  double rmse() const { return rmse_; }
  double num_pixels() const { return num_pixels_; }

 private:
  double per_image_rmse_[kNumImages] = {0};
  double per_image_time_[kNumImages] = {0};
  double per_image_pixels_[kNumImages] = {0};

  double rmse_ = 0;
  double time_ = 0;
  double num_pixels_ = 0;
};

// Computes the RMSE error for two images.
double RMSError(const cv::Mat gt, cv::Mat res) {
  const double num_good_pixels = gt.rows * gt.cols * gt.channels() * 1.0;

  const double norm = cv::norm(gt, res, cv::NORM_L2);
  return std::sqrt(norm * norm / num_good_pixels);
}

}  // namespace benchmark

#endif  // BENCHMARK_UTIL_H_
