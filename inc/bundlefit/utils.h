#ifndef BUNDLEFIT_UTILS_H
#define BUNDLEFIT_UTILS_H

#include <ceres/jet.h>

#include "bundlefit/types.h"

namespace bundlefit::utils {

// Compute difference between two given SE(3) poses
std::pair<double, double> SE3_diff(const Eigen::Matrix3d& rot1, const Eigen::Vector3d& trans1,
                                   const Eigen::Matrix3d& rot2, const Eigen::Vector3d& trans2);

// Compute difference between two given Sim(3) poses
std::tuple<double, double, double> Sim3_diff(
    const Eigen::Matrix3d& rot1, const Eigen::Vector3d& trans1, const double scale1,
    const Eigen::Matrix3d& rot2, const Eigen::Vector3d& trans2, const double scale2);

template <typename T, int N>
double convert_to_double(const ceres::Jet<T, N> val) {
  return val.a;
}
template <typename T>
double convert_to_double(const T val) {
  return static_cast<double>(val);
}

template <typename T>
std::string stringify_dual_number(const T variables[], uint32_t size) {
  std::string str;
  for (uint32_t i = 0; i < size; i++) {
    str += std::to_string(convert_to_double(variables)) + ", ";
  }
  return str;
}

} // namespace bundlefit::utils
#endif // BUNDLEFIT_UTILS_H
