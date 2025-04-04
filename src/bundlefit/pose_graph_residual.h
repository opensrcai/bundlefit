#ifndef BUNDLEFIT_POSE_GRAPH_RESIDUAL_H
#define BUNDLEFIT_POSE_GRAPH_RESIDUAL_H

#include "autodiff_cost_functions.h"
#include "lie/lie_3d.h"

namespace ceres {
namespace internal {
class ResidualBlock;
}
typedef internal::ResidualBlock* ResidualBlockId;
} // namespace ceres
namespace bundlefit {
struct PoseGraphResidual {
  PoseGraphResidual(const Shot* _shot_1, const Shot* _shot_2,
                    const Eigen::Matrix<double, 6, 1>& pose_21)
      : n_dim(6)
      , shot_1(_shot_1)
      , shot_2(_shot_2) {
    assert(n_dim == Shot::NUM_PARAMS[shot_1->type]
           && "The dimensions of relative pose and shot1 did not matched");
    assert(n_dim == Shot::NUM_PARAMS[shot_2->type]
           && "The dimensions of relative pose and shot2 did not matched");
    const auto [R, t] = lie::SE3::exp(pose_21);
    rot_21            = R;
    trans_21          = t;
    scale_21          = 1.0;
  }
  PoseGraphResidual(const Shot* _shot_1, const Shot* _shot_2,
                    const Eigen::Matrix<double, 7, 1>& pose_21)
      : n_dim(7)
      , shot_1(_shot_1)
      , shot_2(_shot_2) {
    assert(n_dim == Shot::NUM_PARAMS[shot_1->type]
           && "The dimensions of relative pose and shot1 did not matched");
    assert(n_dim == Shot::NUM_PARAMS[shot_2->type]
           && "The dimensions of relative pose and shot2 did not matched");
    const auto [R, t, s] = lie::Sim3::exp(pose_21);
    rot_21               = R;
    trans_21             = t;
    scale_21             = s;
  }

  // TODO: implement analytic differentiation
  ceres::CostFunction* generate_cost_function() const {
    auto pose_graph_cost_functor = new PoseGraphCostFunctor(
        shot_1, shot_2, rot_21, trans_21, scale_21, n_dim);
    return pose_graph_cost_functor->generate_cost_function();
  }

  std::vector<double*> stack_parameter_blocks() const {
    return {shot_1->params, shot_2->params};
  }

  const uint32_t n_dim;
  const Shot* shot_1;
  const Shot* shot_2;
  Eigen::Matrix3d rot_21;
  Eigen::Vector3d trans_21;
  double scale_21;
  ceres::ResidualBlockId residual_block_id = nullptr;
};
} // namespace bundlefit

#endif // BUNDLEFIT_POSE_GRAPH_RESIDUAL_H
