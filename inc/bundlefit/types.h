#ifndef BUNDLEFIT_TYPES_H
#define BUNDLEFIT_TYPES_H

#include <Eigen/Core>
#include <unordered_map>
#include <vector>

#include "bundlefit/constexpr_dict.h"

namespace bundlefit {

enum class ObservationType {
  MONOCULAR = 0,
  STEREO,
  DEPTH
};

static constexpr auto NUM_RESIDUALS
    = make_const_dict<ObservationType, uint32_t>(
        {{ObservationType::MONOCULAR, 2},
         {ObservationType::STEREO, 3},
         {ObservationType::DEPTH, 3}});
static constexpr auto MAX_NUM_RESIDUALS = NUM_RESIDUALS.max();

// Eigen vector types
template <int N>
using Vec_t = Eigen::Matrix<double, N, 1>;

} // namespace bundlefit
#endif // BUNDLEFIT_TYPES_H
