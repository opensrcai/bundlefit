#include "lie/lie_group.h"

#include <Eigen/LU>

namespace lie {
double scale_of_sR(const Eigen::Matrix3d& sR) { return std::cbrt(sR.determinant()); }
} // namespace lie
