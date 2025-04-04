#include "bundlefit/utils.h"

#include "lie/lie_3d.h"

namespace bundlefit::utils {

std::pair<double, double> SE3_diff(const Eigen::Matrix3d& rot1, const Eigen::Vector3d& trans1,
                                   const Eigen::Matrix3d& rot2, const Eigen::Vector3d& trans2) {
  const Eigen::Matrix3d dR    = rot1 * rot2.transpose();
  const double r_diff = lie::SO3::log(dR).norm();
  const double t_diff = (trans1 - dR * trans2).norm();
  return {r_diff, t_diff};
}

std::tuple<double, double, double> Sim3_diff(
    const Eigen::Matrix3d& rot1, const Eigen::Vector3d& trans1, const double scale1,
    const Eigen::Matrix3d& rot2, const Eigen::Vector3d& trans2, const double scale2) {
  const Eigen::Matrix3d dR    = rot1 * rot2.transpose();
  const double r_diff = lie::SO3::log(dR).norm();
  const double t_diff = (trans1 - dR * trans2).norm();
  const double s_diff = std::abs(std::log(scale1) - std::log(scale2));
  return {r_diff, t_diff, s_diff};
}

} // namespace bundlefit::utils
