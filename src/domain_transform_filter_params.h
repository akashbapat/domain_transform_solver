#ifndef DOMAIN_TRANSFORM_FILTER_PARAMS_H_
#define DOMAIN_TRANSFORM_FILTER_PARAMS_H_

#include <string>

namespace domain_transform {

template <int N>
struct DomainFilterParamsVec {
  float sigma_x;
  float sigma_y;
  float sigma_r;
  float sigmas[N + 1];
  std::string ToString() const;
};

template <int N>
std::string DomainFilterParamsVec<N>::ToString() const {
  std::string str = "DomainFilterParamsVec<" + std::to_string(N) + ">:\n";
  str += "sigma_x = " + std::to_string(sigma_x) + "\n";
  str += "sigma_y = " + std::to_string(sigma_y) + "\n";
  str += "sigma_r = " + std::to_string(sigma_r) + "\n";

  for (int i = 0; i < N; i++) {
    str += "sigmas[" + std::to_string(i) + "] = " + std::to_string(sigmas[i]) +
           "\n";
  }
  return str;
}

using DomainFilterParams = DomainFilterParamsVec<0>;

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_FILTER_PARAMS_H_
