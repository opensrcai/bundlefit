#ifndef LIE_LIE_3D_H
#define LIE_LIE_3D_H

#include <Eigen/Core>

namespace lie {

template <int N>
using Vec_t = Eigen::Matrix<double, N, 1>;

namespace SO3 {
Eigen::Matrix3d exp(const Eigen::Vector3d& r);
Eigen::Vector3d log(const Eigen::Matrix3d& R);

} // namespace SO3

namespace Sim3 {
class Sim3Cache;
}

namespace SE3 {
std::pair<Eigen::Matrix3d, Eigen::Vector3d> exp(
    const Eigen::Matrix<double, 6, 1>& params);
Eigen::Matrix<double, 6, 1> log(const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& t);
} // namespace SE3

namespace Sim3 {
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> exp(
    const Eigen::Matrix<double, 7, 1>& params);
Eigen::Matrix<double, 7, 1> log(const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& t, const double s);
} // namespace Sim3
} // namespace lie

#endif // LIE_LIE_3D_H
