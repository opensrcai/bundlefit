#ifndef BUNDLEFIT_COST_FUNCTIONS_H
#define BUNDLEFIT_COST_FUNCTIONS_H

#include <ceres/dynamic_autodiff_cost_function.h>

#include "lie/lie_3d_ceres.h"
#include "bundlefit/Sim3.h"
#include "bundlefit/camera.h"
#include "bundlefit/landmark.h"
#include "bundlefit/shot.h"

namespace bundlefit {

// Cost Functor to define reprojection function
struct AutodiffReprojectionCostFunctor {
  AutodiffReprojectionCostFunctor(
      const size_t landmark_index, const Landmark* landmark,
      const size_t lm_lcs_index, const Shot* lm_lcs, const size_t shot_index,
      const Shot* shot, const size_t cam_ext_index, const size_t cam_int_index,
      const Camera* camera, const size_t num_parameter_blocks,
      const double* obs, const double* obs_info, const ObservationType obs_type)
      : landmark_type_(landmark->type)
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
      , obs_(new double[NUM_RESIDUALS[obs_type]])
      , obs_info_(new double[NUM_RESIDUALS[obs_type]])
      , obs_type_(obs_type) {
    for (uint32_t i = 0; i < num_residuals_; i++) {
      obs_[i]      = obs[i];
      obs_info_[i] = obs_info[i];
    }
  }
  ~AutodiffReprojectionCostFunctor() {
    delete[] obs_;
    delete[] obs_info_;
  }
  ceres::CostFunction* generate_cost_function() {
    std::vector<int> param_block_sizes;
    param_block_sizes.resize(num_parameter_blocks_);
    param_block_sizes.at(landmark_index_) = int(num_landmark_params_);
    if (num_lm_lcs_params_ > 0) {
      param_block_sizes.at(lm_lcs_index_) = int(num_lm_lcs_params_);
    }
    param_block_sizes.at(shot_index_) = int(num_shot_params_);
    if (num_cam_ext_params_ > 0) {
      param_block_sizes.at(cam_ext_index_) = int(num_cam_ext_params_);
    }
    param_block_sizes.at(cam_int_index_) = int(num_cam_int_params_);

    typedef AutodiffReprojectionCostFunctor T;

    // Create dynamic cost function (dimension of each parameters should not be
    // compile time constants)
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<T>(this);
    for (const auto& block_size : param_block_sizes) {
      cost_function->AddParameterBlock(block_size);
    }
    cost_function->SetNumResiduals(int(num_residuals_));
    return cost_function;
  }

  template <typename T>
  bool operator()(const T* const* params, T* residuals) const {
    // Data loading
    const T* landmark_params = params[landmark_index_];
    const T* shot_params     = params[shot_index_];
    const T* cam_int_params  = params[cam_int_index_];
    const T* lm_lcs_params
        = num_lm_lcs_params_ != 0 ? params[lm_lcs_index_] : nullptr;
    const T* cam_ext_params
        = num_cam_ext_params_ != 0 ? params[cam_ext_index_] : nullptr;

    T pos_w[Landmark::MAX_NUM_PARAMS];
    if (lm_lcs_type_ == Shot::Type::SE3) {
      // Transform local point to global coordinates with SE3 pose.
      lie::SE3::inv_map(lm_lcs_params, landmark_params, pos_w);
    }
    else if (lm_lcs_type_ == Shot::Type::Sim3) {
      // Transform local point to global coordinates with Sim3 pose.
      lie::Sim3::inv_map(lm_lcs_params, landmark_params, pos_w);
    }
    else if (lm_lcs_type_ == Shot::Type::None) {
      // Landmark is on global coords, do nothing.
      for (uint32_t i = 0; i < 3; i++) {
        pos_w[i] = landmark_params[i];
      }
    }
    else {
      assert(false && "Landmark local coord system has unexpected shot type");
    }

    // Transform world point to local point
    T pos_l[Landmark::MAX_NUM_PARAMS];
    if (shot_type_ == Shot::Type::SE3) {
      lie::SE3::map(shot_params, pos_w, pos_l);
    }
    else if (shot_type_ == Shot::Type::Sim3) {
      lie::Sim3::map(shot_params, pos_w, pos_l);
    }
    else {
      assert(false && "Shot has None type unexpectedly.");
    }

    T pos_c[Landmark::MAX_NUM_PARAMS];
    if (cam_ext_type_ == Camera::ExtrinsicType::SE3) {
      lie::SE3::map(cam_ext_params, pos_l, pos_c);
    }
    else if (cam_ext_type_ == Camera::ExtrinsicType::Sim3) {
      lie::Sim3::map(cam_ext_params, pos_l, pos_c);
    }
    else if (cam_ext_type_ == Camera::ExtrinsicType::None) {
      for (uint32_t i = 0; i < num_landmark_params_; i++) {
        pos_c[i] = pos_l[i];
      }
    }
    else {
      assert(false && "Unexpected camera external parameter type");
    }

    // Reproject local point to image
    T projected_pt[num_residuals_];
    reproject(cam_int_params, cam_model_, pos_c, projected_pt, obs_type_);

    // Compute residual
    for (uint32_t i = 0; i < num_residuals_; i++) {
      residuals[i] = T(obs_info_[i]) * (projected_pt[i] - T(obs_[i]));
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
};

// Cost Functor to define residuals in sim(3) space
struct PoseGraphCostFunctor {
  PoseGraphCostFunctor(const Shot* shot1, const Shot* shot2,
                       const Eigen::Matrix3d& _rot, const Eigen::Vector3d& _trans,
                       const double _scale, const uint32_t _n_dim)
      : n_dim(_n_dim)
      , num_shot1_params(Shot::NUM_PARAMS[shot1->type])
      , num_shot2_params(Shot::NUM_PARAMS[shot2->type])
      , scale(_scale) {
    for (uint32_t i = 0; i < 9; i++) {
      const uint32_t ix = i % 3;
      const uint32_t iy = i / 3;
      rot[i]            = _rot(ix, iy);
    }
    for (uint32_t i = 0; i < 3; i++) {
      trans[i] = _trans(i);
    }
  }

  // TODO: implement analytic diff
  ceres::CostFunction* generate_cost_function() {
    typedef PoseGraphCostFunctor T;

    auto cost_function = new ceres::DynamicAutoDiffCostFunction<T>(this);
    cost_function->AddParameterBlock(int(num_shot1_params));
    cost_function->AddParameterBlock(int(num_shot2_params));
    cost_function->SetNumResiduals(int(n_dim));
    return cost_function;
  }

  const uint32_t n_dim;
  const uint32_t num_shot1_params;
  const uint32_t num_shot2_params;
  double rot[9];
  double trans[3];
  const double scale;

  // Forward function for static autodiff function
  template <typename T>
  bool operator()(const T* const shot1, const T* const shot2, T* residuals,
                  bool* validness = nullptr) const {
    T params1[7];
    T params2[7];
    params1[6] = T(1.0);
    params2[6] = T(1.0);
    for (uint32_t i = 0; i < n_dim; i++) {
      params1[i] = shot1[i];
      params2[i] = shot2[i];
    }
    pgo_core::Sim3_t<T> Sim3_1w(params1);
    pgo_core::Sim3_t<T> Sim3_2w(params2);
    pgo_core::Sim3_t<T> Sim3_21(rot, trans, scale);

    auto Sim3_w2 = Sim3_2w.inverse();

    pgo_core::Sim3_t<T> Sim3_error = Sim3_21 * Sim3_1w * Sim3_w2;

    for (uint32_t i = 0; i < n_dim; i++) {
      residuals[i] = Sim3_error.params[i];
    }
    if (validness) {
      *validness = true;
    }
    return true;
  }

  // Forward function for dynamic autodiff function
  template <typename T>
  bool operator()(const T* const* params, T* residuals) const {
    return (*this)(params[0], params[1], residuals);
  }

  // Compute residual value with squared summation
  // Each parameters are referred from pointers in residual
  // +inf comes back when any computation failure occurred
  double forward(const double* shot1_params, const double* shot2_params) const {
    bool validness = false;
    double residuals[n_dim];
    for (uint32_t i = 0; i < n_dim; i++) residuals[i] = 0;
    (*this)(shot1_params, shot2_params, residuals, &validness);
    if (!validness) {
      return std::numeric_limits<double>::infinity();
    }
    double err = 0;
    for (uint32_t i = 0; i < n_dim; i++) err += residuals[i] * residuals[i];
    return err;
  }
};
} // namespace bundlefit
#endif // BUNDLEFIT_COST_FUNCTIONS_H
