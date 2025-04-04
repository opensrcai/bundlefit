#ifndef BUNDLEFIT_ANALYTIC_COST_FUNCTIONS_H
#define BUNDLEFIT_ANALYTIC_COST_FUNCTIONS_H

#include <ceres/cost_function.h>

#include "bundlefit/camera.h"
#include "bundlefit/landmark.h"
#include "bundlefit/shot.h"
#include "lie/lie_3d_jac.h"

namespace bundlefit {
// Cost function to compute reprojection error with analytic-computed Jacobians
// Limitation: pose parameterization must be SO(3) + R^3 and point must belong
// to R^3 space
class AnalyticReprojectionError : public ceres::CostFunction {
public:
  AnalyticReprojectionError(const size_t landmark_index,
                            const Landmark* landmark, const size_t lm_lcs_index,
                            const Shot* lm_lcs, const size_t shot_index,
                            const Shot* shot, const size_t cam_ext_index,
                            const size_t cam_int_index, const Camera* camera,
                            const size_t num_parameter_blocks,
                            const double* obs, const double* obs_info,
                            const ObservationType obs_type)
      : ceres::CostFunction()
      , landmark_type_(landmark->type)
      , lm_lcs_type_(lm_lcs ? lm_lcs->type : Shot::Type::None)
      , shot_type_(shot->type)
      , cam_ext_type_(camera->has_ext_params ? camera->ext_type
                                             : Camera::ExtrinsicType::None)
      , cam_model_(camera->model)
      , landmark_index_(landmark_index)
      , lm_lcs_index_(lm_lcs_index)
      , shot_index_(shot_index)
      , cam_ext_index_(cam_ext_index)
      , cam_int_index_(cam_int_index)
      , num_parameter_blocks_(num_parameter_blocks)
      , num_landmark_params_(Landmark::NUM_PARAMS[landmark->type])
      , num_lm_lcs_params_(lm_lcs ? Shot::NUM_PARAMS[lm_lcs->type] : 0)
      , num_shot_params_(Shot::NUM_PARAMS[shot->type])
      , num_cam_ext_params_(camera->has_ext_params
                                ? Camera::NUM_EXTRINSIC_PARAMS[camera->ext_type]
                                : 0)
      , num_cam_int_params_(Camera::NUM_INTRINSIC_PARAMS[camera->model])
      , num_residuals_(NUM_RESIDUALS[obs_type])
      , obs_(new double[MAX_NUM_RESIDUALS])
      , obs_info_(new double[MAX_NUM_RESIDUALS])
      , obs_type_(obs_type)
      , shot_cache_(shot->cache.get()) {
    for (uint32_t i = 0; i < num_residuals_; i++) {
      obs_[i]      = obs[i];
      obs_info_[i] = obs_info[i];
    }
    set_num_residuals(int(num_residuals_));

    mutable_parameter_block_sizes()->resize(num_parameter_blocks_);
    mutable_parameter_block_sizes()->at(landmark_index_)
        = int(num_landmark_params_);
    if (lm_lcs) {
      mutable_parameter_block_sizes()->at(lm_lcs_index_)
          = int(num_lm_lcs_params_);
    }
    mutable_parameter_block_sizes()->at(shot_index_) = int(num_shot_params_);
    if (camera->has_ext_params) {
      mutable_parameter_block_sizes()->at(cam_ext_index_)
          = int(num_cam_ext_params_);
    }
    mutable_parameter_block_sizes()->at(cam_int_index_)
        = int(num_cam_int_params_);
  }
  ~AnalyticReprojectionError() override;

  // Compute residual with Jacobians
  bool Evaluate(const double* const* x, double* residuals,
                double** jacobians) const override {
    // 0. Input data preparation

    // Primal values
    const double* landmark_params = x[landmark_index_];
    const double* shot_params     = x[shot_index_];
    const double* cam_int_params  = x[cam_int_index_];
    const double* lm_lcs_params
        = num_lm_lcs_params_ != 0 ? x[lm_lcs_index_] : nullptr;
    const double* cam_ext_params
        = num_cam_ext_params_ != 0 ? x[cam_ext_index_] : nullptr;

    // Jacobians
    double* J_landmark = jacobians ? jacobians[landmark_index_] : nullptr;
    double* J_shot     = jacobians ? jacobians[shot_index_] : nullptr;
    double* J_cam_int  = jacobians ? jacobians[cam_int_index_] : nullptr;
    double* J_lm_lcs   = jacobians && num_lm_lcs_params_ != 0
                             ? jacobians[lm_lcs_index_]
                             : nullptr;
    double* J_cam_ext  = jacobians && num_cam_ext_params_ != 0
                             ? jacobians[cam_ext_index_]
                             : nullptr;

    // 1. Forward primal tracing with Jacobian

    // Transform landmark to world coordinate
    // Do nothing when landmark is in world coordinate
    double pos_w[Landmark::MAX_NUM_PARAMS];
    double dpw_dlcs[Landmark::MAX_NUM_PARAMS][Shot::MAX_NUM_PARAMS];
    double dpw_dlm[Landmark::MAX_NUM_PARAMS][Landmark::MAX_NUM_PARAMS];
    bool no_pw_derivation = false;
    if (lm_lcs_type_ == Shot::Type::SE3) {
      // Transform local point to global coordinates with SE3 pose.
      lie::SE3::inv_map(lm_lcs_params, landmark_params, pos_w, J_lm_lcs,
                        dpw_dlcs, J_landmark, dpw_dlm);
    }
    else if (lm_lcs_type_ == Shot::Type::Sim3) {
      // Transform local point to global coordinates with Sim3 pose.
      lie::Sim3::inv_map(lm_lcs_params, landmark_params, pos_w, J_lm_lcs,
                         dpw_dlcs, J_landmark, dpw_dlm);
    }
    else if (lm_lcs_type_ == Shot::Type::None) {
      // Landmark is on global coords, do nothing.
      for (uint32_t i = 0; i < num_landmark_params_; i++) {
        pos_w[i] = landmark_params[i];
      }
      no_pw_derivation = true;
    }
    else {
      assert(false && "Landmark local coord system has unexpected shot type");
    }

    // Transform point in world to local coordinates
    // Jacobians ∂pl / ∂s, ∂pl / ∂pw are computed if required
    //     pl : point in local coordinates (R^3)
    //     s : shot parameters, so(3) and R^3
    //     pw : point in world coordinates (R^3)
    double pos_l[Landmark::MAX_NUM_PARAMS];
    double dpl_ds[Landmark::MAX_NUM_PARAMS][Shot::MAX_NUM_PARAMS];
    double dpl_dpw[Landmark::MAX_NUM_PARAMS][Landmark::MAX_NUM_PARAMS];
    if (shot_type_ == Shot::Type::SE3) {
      lie::SE3::map(shot_params, pos_w, pos_l, J_shot, dpl_ds,
                    J_lm_lcs || J_landmark, dpl_dpw, shot_cache_);
    }
    else if (shot_type_ == Shot::Type::Sim3) {
      lie::Sim3::map(shot_params, pos_w, pos_l, J_shot, dpl_ds,
                     J_lm_lcs || J_landmark, dpl_dpw, shot_cache_);
    }
    else {
      assert(false && "Shot has None type unexpectedly.");
    }

    // Compute camera coordinate of landmark if the camera have offset (= camera
    // pose is not identical to local coordinate) When the camera is in the
    // center of local coordinate, do nothing
    double pos_c[Landmark::MAX_NUM_PARAMS];
    double dpc_dpl[Landmark::MAX_NUM_PARAMS][Landmark::MAX_NUM_PARAMS];
    double dpc_dpe[Landmark::MAX_NUM_PARAMS][Shot::MAX_NUM_PARAMS];
    bool no_pc_derivation = false;
    if (cam_ext_type_ == Camera::ExtrinsicType::SE3) {
      lie::SE3::map(cam_ext_params, pos_l, pos_c, J_cam_ext, dpc_dpe,
                    J_shot || J_lm_lcs || J_landmark, dpc_dpl);
    }
    else if (cam_ext_type_ == Camera::ExtrinsicType::Sim3) {
      lie::Sim3::map(cam_ext_params, pos_l, pos_c, J_cam_ext, dpc_dpe,
                     J_shot || J_lm_lcs || J_landmark, dpc_dpl);
    }
    else if (cam_ext_type_ == Camera::ExtrinsicType::None) {
      for (size_t i = 0; i < num_landmark_params_; i++) {
        pos_c[i] = pos_l[i];
      }
      no_pc_derivation = true;
    }
    else {
      assert(false && "Unexpected camera external parameter type");
    }

    // Reproject point in local coordinates to image
    // Jacobians ∂i / ∂c, ∂i / ∂pl are computed if required
    //     i : image coordinates (R^[n_dim], n_dim is 2(monocular) or 3(stereo))
    //     c : camera parameters
    double projected_pt[MAX_NUM_RESIDUALS];
    double di_dc[MAX_NUM_RESIDUALS][Camera::MAX_NUM_INT_PARAMS];
    double di_dpc[MAX_NUM_RESIDUALS][Landmark::MAX_NUM_PARAMS];
    reproject(cam_int_params, cam_model_, pos_c, projected_pt, obs_type_,
              J_cam_int, J_cam_ext || J_shot || J_lm_lcs || J_landmark, di_dc,
              di_dpc);

    // Finally, reprojection error has been computed
    for (uint32_t i = 0; i < num_residuals_; i++) {
      residuals[i] = obs_info_[i] * (projected_pt[i] - obs_[i]);
    }

    double do_dc[MAX_NUM_RESIDUALS][Camera::MAX_NUM_INT_PARAMS];
    double do_dpc[MAX_NUM_RESIDUALS][Landmark::MAX_NUM_PARAMS];
    for (uint32_t i = 0; i < num_residuals_; i++) {
      for (uint32_t j = 0; j < num_cam_int_params_; j++) {
        do_dc[i][j] = obs_info_[i] * di_dc[i][j];
      }
      for (uint32_t j = 0; j < num_landmark_params_; j++) {
        do_dpc[i][j] = obs_info_[i] * di_dpc[i][j];
      }
    }

    // 2. Reverse adjoint traceing

    // Store Jacobian from camera intrinsic parameters to image coordinates ∂i /
    // ∂ci
    if (J_cam_int) {
      for (uint32_t i = 0; i < num_residuals_; i++) {
        for (uint32_t j = 0; j < num_cam_int_params_; j++) {
          J_cam_int[i * num_cam_int_params_ + j] = do_dc[i][j];
        }
      }
    }

    // Store Jacobian from camera extrinsic parameters to image coordinates ∂i /
    // ∂ce
    if (J_cam_ext) {
      for (uint32_t i = 0; i < num_residuals_; i++) {
        for (uint32_t j = 0; j < num_cam_ext_params_; j++) {
          J_cam_ext[i * num_cam_ext_params_ + j]
              = (do_dpc[i][0] * dpc_dpe[0][j] + do_dpc[i][1] * dpc_dpe[1][j]
                 + do_dpc[i][2] * dpc_dpe[2][j]);
        }
      }
    }

    if (J_shot || J_lm_lcs || J_landmark) {
      double do_dpl[MAX_NUM_RESIDUALS][Landmark::MAX_NUM_PARAMS];
      if (no_pc_derivation) {
        for (uint32_t i = 0; i < num_residuals_; i++) {
          for (uint32_t j = 0; j < num_landmark_params_; j++) {
            do_dpl[i][j] = do_dpc[i][j];
          }
        }
      }
      else {
        for (uint32_t i = 0; i < num_residuals_; i++) {
          for (uint32_t j = 0; j < num_landmark_params_; j++) {
            do_dpl[i][j] = do_dpc[i][0] * dpc_dpl[0][j]
                           + do_dpc[i][1] * dpc_dpl[1][j]
                           + do_dpc[i][2] * dpc_dpl[2][j];
          }
        }
      }

      // Store Jacobian from shot parameters to image coordinates ∂i / ∂s
      if (J_shot) {
        for (uint32_t i = 0; i < num_residuals_; i++) {
          for (uint32_t j = 0; j < num_shot_params_; j++) {
            J_shot[j + num_shot_params_ * i] = do_dpl[i][0] * dpl_ds[0][j]
                                               + do_dpl[i][1] * dpl_ds[1][j]
                                               + do_dpl[i][2] * dpl_ds[2][j];
          }
        }
      }

      if (J_lm_lcs || J_landmark) {
        double do_dpw[MAX_NUM_RESIDUALS][Landmark::MAX_NUM_PARAMS];
        for (uint32_t i = 0; i < num_residuals_; i++) {
          for (uint32_t j = 0; j < num_landmark_params_; j++) {
            do_dpw[i][j] = do_dpl[i][0] * dpl_dpw[0][j]
                           + do_dpl[i][1] * dpl_dpw[1][j]
                           + do_dpl[i][2] * dpl_dpw[2][j];
          }
        }

        // Store Jacobian from local coordynate system parameters to image
        // coordinates ∂i / ∂lcs
        if (J_lm_lcs) {
          for (uint32_t i = 0; i < num_residuals_; i++) {
            for (uint32_t j = 0; j < num_lm_lcs_params_; j++) {
              J_lm_lcs[j + num_lm_lcs_params_ * i]
                  = do_dpw[i][0] * dpw_dlcs[0][j]
                    + do_dpw[i][1] * dpw_dlcs[1][j]
                    + do_dpw[i][2] * dpw_dlcs[2][j];
            }
          }
        }

        // Store Jacobian from landmark to image coordinates ∂i / ∂lm
        if (J_landmark) {
          if (no_pw_derivation) {
            for (uint32_t i = 0; i < num_residuals_; i++) {
              for (uint32_t j = 0; j < num_landmark_params_; j++) {
                J_landmark[j + 3 * i] = do_dpw[i][j];
              }
            }
          }
          else {
            for (uint32_t i = 0; i < num_residuals_; i++) {
              for (uint32_t j = 0; j < num_landmark_params_; j++) {
                J_landmark[j + 3 * i] = do_dpw[i][0] * dpw_dlm[0][j]
                                        + do_dpw[i][1] * dpw_dlm[1][j]
                                        + do_dpw[i][2] * dpw_dlm[2][j];
              }
            }
          }
        }
      }
    }

    return true;
  }

  const Landmark::Type landmark_type_;
  const Shot::Type lm_lcs_type_;
  const Shot::Type shot_type_;
  const Camera::ExtrinsicType cam_ext_type_;
  const Camera::ProjectionModel cam_model_;

private:
  const size_t landmark_index_;
  const size_t lm_lcs_index_;
  const size_t shot_index_;
  const size_t cam_ext_index_;
  const size_t cam_int_index_;
  const size_t num_parameter_blocks_;

  const uint32_t num_landmark_params_;
  const uint32_t num_lm_lcs_params_;
  const uint32_t num_shot_params_;
  const uint32_t num_cam_ext_params_;
  const uint32_t num_cam_int_params_;
  const uint32_t num_residuals_;

  double* obs_;
  double* obs_info_;
  const ObservationType obs_type_;

  const lie::Sim3::Sim3Cache* shot_cache_;
};

AnalyticReprojectionError::~AnalyticReprojectionError() {
  delete[] obs_;
  delete[] obs_info_;
}
} // namespace bundlefit
#endif // BUNDLEFIT_ANALYTIC_COST_FUNCTIONS_H
