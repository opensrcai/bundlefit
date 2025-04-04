#include <lie/lie_3d.h>

#include <Eigen/Geometry>

namespace lie {
namespace SO3 {
Eigen::Matrix3d exp(const Eigen::Vector3d& r) {
  return Eigen::AngleAxisd(r.norm(), r.normalized()).toRotationMatrix();
}
Eigen::Vector3d log(const Eigen::Matrix3d& R) {
  const Eigen::AngleAxisd angle_axis(R);
  return angle_axis.axis() * angle_axis.angle();
}
} // namespace SO3
namespace SE3 {
std::pair<Eigen::Matrix3d, Eigen::Vector3d> exp(const Eigen::Matrix<double, 6, 1>& params) {
  return {SO3::exp(params.topRows(3)), params.bottomRows(3)};
}
Eigen::Matrix<double, 6, 1> log(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  Eigen::Matrix<double, 6, 1> params;
  params.topRows(3)    = SO3::log(R);
  params.bottomRows(3) = t;
  return params;
}
} // namespace SE3
namespace Sim3 {
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> exp(const Eigen::Matrix<double, 7, 1>& params) {
  return {SO3::exp(params.topRows(3)), params.middleRows(3, 3),
          std::exp(params(6))};
}
Eigen::Matrix<double, 7, 1> log(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double s) {
  Eigen::Matrix<double, 7, 1> params;
  params.topRows(3)       = SO3::log(R);
  params.middleRows(3, 3) = t;
  params(6)               = std::log(s);
  return params;
}
} // namespace Sim3
} // namespace lie
