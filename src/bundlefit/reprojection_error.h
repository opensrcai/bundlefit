#ifndef BUNDLEFIT_REPROJECTION_ERROR_H
#define BUNDLEFIT_REPROJECTION_ERROR_H

#include <ceres/loss_function.h>

#include "bundlefit/analytic_cost_functions.h"
#include "bundlefit/autodiff_cost_functions.h"
#include "bundlefit/bundle_adjuster.h"
#include "bundlefit/camera.h"
#include "bundlefit/landmark.h"
#include "bundlefit/shot.h"

namespace ceres {
namespace internal {
class ResidualBlock;
}
typedef internal::ResidualBlock* ResidualBlockId;
} // namespace ceres

namespace bundlefit {

// Struct for reprojection error in BundleAdjuster class
struct ReprojectionError {
  ReprojectionError() = delete;

  template <int N>
  ReprojectionError(const Camera* _camera, const Shot* _shot,
                    const Landmark* _landmark, const Vec_t<N>& _obs,
                    const Vec_t<N> _obs_info, const ObservationType _obs_type,
                    const LossType _loss_type, const double _loss_scale)
      : camera(_camera)
      , shot(_shot)
      , landmark(_landmark)
      , n_dim(NUM_RESIDUALS[_obs_type])
      , obs(new double[MAX_NUM_RESIDUALS])
      , obs_info(new double[MAX_NUM_RESIDUALS])
      , obs_type(_obs_type)
      , loss_type(_loss_type)
      , loss_scale(_loss_scale) {
    assert(N == n_dim && "Different size of observation given");
    for (uint32_t i = 0; i < n_dim; i++) {
      obs[i]      = _obs[i];
      obs_info[i] = _obs_info[i];
    }

    num_parameter_blocks_ = 0;
    landmark_index_       = num_parameter_blocks_++;
    if (_landmark->local_coord_system) {
      lm_lcs_index_ = num_parameter_blocks_++;
    }
    shot_index_ = num_parameter_blocks_++;
    if (_camera->has_ext_params) {
      cam_ext_index_ = num_parameter_blocks_++;
    }
    cam_int_index_ = num_parameter_blocks_++;
  }
  ~ReprojectionError() {
    delete[] obs;
    delete[] obs_info;
  }

  bool is_analytic_diff_available() const {
    return (shot->type == Shot::Type::SE3 || shot->type == Shot::Type::Sim3)
           && landmark->type == Landmark::Type::Point3D;
  }

  ceres::CostFunction* generate_cost_function(bool allow_analytic_diff) const {
    if (allow_analytic_diff && is_analytic_diff_available()) {
      return new AnalyticReprojectionError(
          landmark_index_, landmark, lm_lcs_index_,
          landmark->local_coord_system, shot_index_, shot, cam_ext_index_,
          cam_int_index_, camera, num_parameter_blocks_, obs, obs_info,
          obs_type);
    }
    else {
      auto autodiff_cost_functor = new AutodiffReprojectionCostFunctor(
          landmark_index_, landmark, lm_lcs_index_,
          landmark->local_coord_system, shot_index_, shot, cam_ext_index_,
          cam_int_index_, camera, num_parameter_blocks_, obs, obs_info,
          obs_type);
      return autodiff_cost_functor->generate_cost_function();
    }
  }

  std::vector<double*> stack_parameter_blocks() const {
    std::vector<double*> param_blocks(num_parameter_blocks_);
    param_blocks.at(landmark_index_) = landmark->params;
    param_blocks.at(shot_index_)     = shot->params;
    param_blocks.at(cam_int_index_)  = camera->int_params;

    if (landmark->local_coord_system) {
      param_blocks.at(lm_lcs_index_) = landmark->local_coord_system->params;
    }
    if (camera->has_ext_params) {
      param_blocks.at(cam_ext_index_) = camera->ext_params;
    }
    return param_blocks;
  }

  const Camera* camera;
  const Shot* shot;
  const Landmark* landmark;

  const uint32_t n_dim;
  double* obs;
  double* obs_info;

  const ObservationType obs_type;

  size_t landmark_index_;
  size_t lm_lcs_index_;
  size_t shot_index_;
  size_t cam_ext_index_;
  size_t cam_int_index_;
  size_t num_parameter_blocks_;

  ceres::ResidualBlockId residual_block_id = nullptr;

  LossType loss_type = LossType::TRIVIAL;
  double loss_scale  = 1.0;
};

} // namespace bundlefit

#endif // BUNDLEFIT_REPROJECTION_ERROR_H
