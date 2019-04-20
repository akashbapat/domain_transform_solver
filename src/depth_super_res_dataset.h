#ifndef DEPTH_SUPER_RES_DATASET_H_
#define DEPTH_SUPER_RES_DATASET_H_

#include <cmath>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "domain_transform_common.h"
#include "domain_transform_image_io.h"
#include "edge_aware_solver.h"

namespace benchmark {
// Defines a class to load, preprocess and write depth super resolution dataset
// shared by Jon Barron at
// https://drive.google.com/file/d/0B4nuwEMaEsnmaDI3bm5VeDRxams/view .
class DepthSuperResData {
 public:
  // There are twelve images in the dataset, change this if you add more.
  static constexpr int kNumImages = 12;

  DepthSuperResData(const std::string& path_to_data_top_dir,
                    const std::string& algorithm_init_name)
      : path_to_data_top_dir_(path_to_data_top_dir),
        algorithm_init_name_(algorithm_init_name) {}

  // Defines the maximum size of the image in the entire dataset.
  domain_transform::ImageDim MaxImageDim() const {
    domain_transform::ImageDim max_image_dims;
    max_image_dims.width = 1536;
    max_image_dims.height = 1536;
    return max_image_dims;
  }

  // Reads the next image in the dataset. If there are no more images, this
  // starts again at the beginning.
  domain_transform::ProblemImageData GetProblemImageData(int* upsample_level) {
    domain_transform::ProblemImageData data;
    const std::string color_image_path =
        path_to_data_top_dir_ + "/_input/" + folder_names_[folder_idx_] + "/" +
        folder_names_[folder_idx_] + "_color.png";
    data.rgba_image = domain_transform::ReadRGBAImageFrom(color_image_path);

    constexpr float kDisparityScale = 1.0f / (256.0 * 256.0 - 1);
    const std::string target_full_path =
        path_to_data_top_dir_ + "/" + algorithm_init_name_ + "/" +
        folder_names_[folder_idx_] + "/" + image_names_[res_idx_] + ".png";

    data.target = domain_transform::ReadFloatImageFrom16bit(target_full_path,
                                                            kDisparityScale);

    const std::string gt_full_path = path_to_data_top_dir_ + "/" + "_gth/" +
                                     folder_names_[folder_idx_] + ".png";

    data.gt = cv::imread(gt_full_path, CV_LOAD_IMAGE_GRAYSCALE);

    // Read the gaussian bump weights.
    const std::string confidence_weight_full_path =
        path_to_data_top_dir_ + "/weights/" + folder_names_[folder_idx_] + "/" +
        conf_names_[res_idx_] + ".png";
    const cv::Mat confidence_bump_map =
        cv::imread(confidence_weight_full_path, CV_LOAD_IMAGE_ANYDEPTH);

    constexpr float kConfidenceScale = 1.0f / (256.0 * 256.0 - 1);
    cv::Mat float_confidence_bump_map =
        domain_transform::ReadFloatImageFrom16bit(confidence_weight_full_path,
                                                  kConfidenceScale);

    double min_val, max_val;
    cv::minMaxLoc(float_confidence_bump_map, &min_val, &max_val);
    data.confidence = float_confidence_bump_map / max_val;

    *upsample_level = std::pow(2, res_idx_ + 1);
    res_idx_++;

    if (res_idx_ >= image_names_.size()) {
      res_idx_ = 0;
      folder_idx_++;
    }

    folder_idx_ = folder_idx_ % static_cast<int>(folder_names_.size());

    return data;
  }

  // Assumes that the directory structure with the algorithm already exists.
  void SaveResultForAlogrithm(const std::string& algo_name,
                              const cv::Mat& result) {
    int prev_folder_idx = folder_idx_;
    int prev_res_idx = res_idx_;
    if (res_idx_ == 0) {
      prev_folder_idx--;
    }

    prev_folder_idx = prev_folder_idx % static_cast<int>(folder_names_.size());

    // Add size to avoid negative indices.
    if (prev_folder_idx < 0) {
      prev_folder_idx += folder_names_.size();
    }

    prev_res_idx--;
    prev_res_idx = prev_res_idx % static_cast<int>(image_names_.size());

    // Add size to avoid negative indices.
    if (prev_res_idx < 0) {
      prev_res_idx += image_names_.size();
    }

    const std::string result_path = path_to_data_top_dir_ + "/" + algo_name +
                                    "/" + folder_names_[prev_folder_idx] + "/" +
                                    image_names_[prev_res_idx] + ".png";

    cv::imwrite(result_path, result);
  }

 private:
  const std::vector<std::string> folder_names_ = {"art_big", "books_big",
                                                  "moebius_big"};

  const std::vector<std::string> image_names_ = {"output_1_n", "output_2_n",
                                                 "output_3_n", "output_4_n"};

  const std::vector<std::string> conf_names_ = {"weight_1_n", "weight_2_n",
                                                "weight_3_n", "weight_4_n"};

  std::string path_to_data_top_dir_;
  std::string algorithm_init_name_;
  int folder_idx_ = 0;
  int res_idx_ = 0;
};

}  // namespace benchmark

#endif  //  DEPTH_SUPER_RES_DATASET_H_
