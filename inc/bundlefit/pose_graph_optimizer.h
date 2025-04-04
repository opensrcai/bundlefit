#ifndef BUNDLEFIT_GRAPH_OPTIMIZER_IMPL_H
#define BUNDLEFIT_GRAPH_OPTIMIZER_IMPL_H

#include <ceres/problem.h>

#include <list>

#include "bundlefit/types.h"

namespace ceres {
namespace internal {
class ResidualBlock;
}
typedef internal::ResidualBlock* ResidualBlockId;
} // namespace ceres

namespace bundlefit {

struct Shot;
struct PoseGraphResidual;

class PoseGraphOptimizer {
public:
  PoseGraphOptimizer(const uint32_t num_threads = 2,
                     const bool use_Sim3_cache  = true);
  ~PoseGraphOptimizer();

  void add_SE3_shot(const uint32_t id, const Eigen::Matrix<double, 6, 1>& params,
                    const bool is_fixed = false);
  Eigen::Matrix<double, 6, 1> get_SE3_shot_pose(const uint32_t id) const;

  void add_Sim3_shot(const uint32_t id, const Eigen::Matrix<double, 7, 1>& params,
                     const bool is_fixed = false);
  Eigen::Matrix<double, 7, 1> get_Sim3_shot_pose(const uint32_t id) const;

  Eigen::Matrix<double, 6, 1> get_shot_pose_as_SE3(const uint32_t id) const;

  void add_pose_graph_residual(const uint32_t shot1_id, const uint32_t shot2_id,
                               const Eigen::Matrix<double, 6, 1>& pose_21);
  void add_pose_graph_residual(const uint32_t shot1_id, const uint32_t shot2_id,
                               const Eigen::Matrix<double, 7, 1>& pose_21);

  void construct_problem();
  void fit(const uint32_t iterations, const bool dump_progress = false,
           const bool dump_full_progress = false);

  void evaluate();

private:
  ceres::Problem* problem_;
  const uint32_t num_threads_;
  // TODO: use analytic diff
  const bool use_Sim3_cache_;

  std::unordered_map<uint32_t, Shot*> shots_;
  std::list<PoseGraphResidual*> residuals_;
};
} // namespace bundlefit
#endif // BUNDLEFIT_GRAPH_OPTIMIZER_IMPL_H
