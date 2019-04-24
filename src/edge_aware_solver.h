#ifndef EDGE_AWARE_SOLVER_H_
#define EDGE_AWARE_SOLVER_H_

#include <memory>

#include "domain_transform_solver.h"
#include "error_types.h"
#include "hfbs.h"

namespace domain_transform {
// DtsOptions assembles the options typically useful for domain transform solver
// at a single place.
struct DtsOptions {
  DomainFilterParams filter_options;
  DomainOptimizeParams optim_options;
  DomainFilterParamsVec<1> conf_params;
  COLOR_SPACE color_space;
  ImageDim image_dims;
  float overwrite_target_above_conf = -1;
};

// HfbsOptions assembles the options typically useful for HFBS at a single
// place.
struct HfbsOptions {
  HFBSGridParams grid_params;
  domain_transform::COLOR_SPACE color_space;
  ImageDim image_dims;
  float overwrite_target_above_conf = -1;
};

// Template function signature which is specialized for each edge-aware solver.
template <typename SolverOptions, typename SolverType>
void Solve(const SolverOptions& solver_options, SolverType* solver);

// Template specialization DTS to solve the problem given the options.
template <>
void Solve(const DtsOptions& solver_options,
           domain_transform::DomainTransformSolver* solver) {
  solver->ComputeColorSpaceDifferential(solver_options.filter_options);
  solver->IntegrateColorDifferentials();

  solver->Optimize(solver_options.optim_options,
                   solver_options.overwrite_target_above_conf);
}

// Template specialization HFBS to solve the problem given the options.
template <>
void Solve(const HfbsOptions& solver_options, HFBS* solver) {
  solver->Optimize();
}

// Common function to solve an edge-aware optimization problem which all
// algorithms should use. This also restricts the interface of the solver.
template <typename SolverOptions, typename SolverType>
float EdgeAwareOptimize(const cv::Mat& rgba_image, const cv::Mat& target_image,
                        const cv::Mat& confidence,
                        const SolverOptions& solver_options, SolverType* solver,
                        cv::Mat* optim_image) {
  solver->SetImageDim(solver_options.image_dims);

  solver->ClearAll();

  solver->InitFrame(solver_options.color_space, rgba_image.data,
                    reinterpret_cast<float*>(target_image.data),
                    reinterpret_cast<float*>(confidence.data));

  // Start Logging time.
  cudaEvent_t start, stop;
  GPU_CHECK(cudaEventCreate(&start));
  GPU_CHECK(cudaEventCreate(&stop));
  GPU_CHECK(cudaEventRecord(start, 0));

  Solve(solver_options, solver);

  // Stop logging time.
  GPU_CHECK(cudaEventRecord(stop, 0));
  GPU_CHECK(cudaEventSynchronize(stop));
  float local_time_in_ms = 0;
  GPU_CHECK(cudaEventElapsedTime(&local_time_in_ms, start, stop));

  solver->Download(ImageType::OPTIMIZED_QUANTITY,
                   reinterpret_cast<float*>((*optim_image).data));

  return local_time_in_ms;
}

// Image data required as an input for edge-aware optimization.
struct ProblemImageData {
  cv::Mat rgba_image;
  cv::Mat target;
  cv::Mat confidence;
  cv::Mat gt;
};

// A interface class to define an edge-aware solver to handle different kinds of
// datasets, applications and solvers.
template <typename SolverType, typename SolverOptionsType>
class EdgeAwareSolver {
 public:
  EdgeAwareSolver() {}

  virtual void PreProcess(cv::Mat* rgba_image, cv::Mat* target_image,
                          cv::Mat* confidence) {}
  virtual void PostProcessResult(cv::Mat* optimized_result) {}

  SolverType& solver() { return *solver_; }

  const SolverOptionsType& options() { return solver_options_; }

 protected:
  std::unique_ptr<SolverType> solver_;
  SolverOptionsType solver_options_;
};

}  //  namespace domain_transform

#endif  // EDGE_AWARE_SOLVER_H_
