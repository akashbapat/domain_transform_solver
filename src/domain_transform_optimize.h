#ifndef DOMAIN_TRANSFORM_OPTIMIZE_H_
#define DOMAIN_TRANSFORM_OPTIMIZE_H_

#include <string>

namespace domain_transform {

enum class RobustLoss { L1, L2, CHARBONNIER };

std::string RobustLossToString(const RobustLoss& loss) {
  std::string str = "RobustLoss::";
  switch (loss) {
    case RobustLoss::L2: {
      str += "L2";
      break;
    }
    case RobustLoss::L1: {
      str += "L1";
      break;
    }
    case RobustLoss::CHARBONNIER: {
      str += "CHARBONNIER";
      break;
    }
  }
  return str;
}

struct DomainOptimizeParams {
  int num_iterations = 3;
  RobustLoss loss = RobustLoss::L2;
  float lambda = 1.0f;
  float sigma_z = 100;
  float step_size = 0.99;

  // Additional lambdas.
  float photo_consis_lambda = 1.0f;
  float DT_lambda = 0.2f;

  std::string ToString() const {
    std::string str = "Num iter: " + std::to_string(num_iterations) + "\n";
    str += "loss: " + RobustLossToString(loss) + "\n";
    str += "lambda: " + std::to_string(lambda) + "\n";
    str += "sigma_z: " + std::to_string(sigma_z) + "\n";
    str += "step_size: " + std::to_string(step_size) + "\n";
    str += "photo_consis_lambda: " + std::to_string(photo_consis_lambda) + "\n";
    str += "DT_lambda: " + std::to_string(DT_lambda) + "\n";
    return str;
  }
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_OPTIMIZE_H_
