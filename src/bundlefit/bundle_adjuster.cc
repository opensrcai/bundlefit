#include "bundlefit/bundle_adjuster.h"

#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <Eigen/LU>

#include "bundlefit/camera.h"
#include "bundlefit/ceres.h"
#include "bundlefit/landmark.h"
#include "bundlefit/reprojection_error.h"
#include "bundlefit/shot.h"

namespace bundlefit {

static ceres::LossFunction* CreateLossFunction(const LossType loss_type,
                                               const double scale) {
  switch (loss_type) {
    case LossType::TRIVIAL:
      return new ceres::TrivialLoss();
    case LossType::HUBER_LOSS:
      return new ceres::HuberLoss(scale);
    case LossType::SOFT_L1:
      return new ceres::SoftLOneLoss(scale);
    case LossType::CAUCHY:
      return new ceres::CauchyLoss(scale);
  }
  assert(false);
  // TODO: error handling
  return nullptr;
}

class BAIterationCallback : public ceres::IterationCallback {
public:
  explicit BAIterationCallback(BundleAdjuster& _ba)
      : ba(_ba) {}
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& /*summary*/) override;
  BundleAdjuster& ba;
};
ceres::CallbackReturnType BAIterationCallback::operator()(
    const ceres::IterationSummary&) {
  if (ba.break_required_) {
    return ceres::SOLVER_ABORT;
  }
  ba.clear_Sim3_cache();
  return ceres::SOLVER_CONTINUE;
}

BundleAdjuster::BundleAdjuster(const uint32_t num_threads,
                               const bool use_analytic_diff,
                               const bool use_Sim3_cache)
    : problem_(nullptr)
    , num_threads_(num_threads)
    , use_analytic_diff_(use_analytic_diff)
    , use_Sim3_cache_(use_Sim3_cache) {}

BundleAdjuster::~BundleAdjuster() {
  for (auto error : reprojection_errors_) {
    delete error;
  }
  for (auto landmark : landmarks_) {
    delete landmark.second;
  }
  for (auto shot : shots_) {
    delete shot.second;
  }
  for (auto camera : cameras_) {
    delete camera.second;
  }
}

// Add perspective camera to BundleAdjuster
void BundleAdjuster::add_perspective_camera(const uint32_t id, const double fx,
                                            const double fy, const double cx,
                                            const double cy, const int flags) {
  const bool is_int_fixed = flags & FLAG_FIX_INTRINSIC_PARAMS;
  cameras_[id] = new PerspectiveCamera(id, fx, fy, cx, cy, is_int_fixed);
}
void BundleAdjuster::add_perspective_camera(const uint32_t id, const double fx,
                                            const double fy, const double cx,
                                            const double cy,
                                            double pixel_baseline,
                                            const int flags) {
  const bool is_int_fixed = flags & FLAG_FIX_INTRINSIC_PARAMS;
  cameras_[id]
      = new PerspectiveCamera(id, fx, fy, cx, cy, pixel_baseline, is_int_fixed);
}
void BundleAdjuster::add_perspective_camera(const uint32_t id, const double fx,
                                            const double fy, const double cx,
                                            const double cy,
                                            const Eigen::Matrix<double, 6, 1>& pose_offset,
                                            const int flags) {
  const bool is_int_fixed = flags & FLAG_FIX_INTRINSIC_PARAMS;
  const bool is_ext_fixed = flags & FLAG_FIX_EXTRINSIC_PARAMS;
  cameras_[id] = new PerspectiveCamera(id, fx, fy, cx, cy, is_int_fixed,
                                       pose_offset, is_ext_fixed);
}

// Add equirectangular camera to BundleAdjuster
void BundleAdjuster::add_equirectangular_camera(const uint32_t id,
                                                uint32_t rows, uint32_t cols,
                                                const int flags) {
  const bool is_int_fixed = flags & FLAG_FIX_INTRINSIC_PARAMS;
  auto camera  = new EquirectangularCamera(id, rows, cols, is_int_fixed);
  cameras_[id] = camera;
}
void BundleAdjuster::add_equirectangular_camera(const uint32_t id,
                                                uint32_t rows, uint32_t cols,
                                                const Eigen::Matrix<double, 6, 1>& pose_offset,
                                                const int flags) {
  const bool is_int_fixed = flags & FLAG_FIX_INTRINSIC_PARAMS;
  const bool is_ext_fixed = flags & FLAG_FIX_EXTRINSIC_PARAMS;
  auto camera  = new EquirectangularCamera(id, rows, cols, is_int_fixed,
                                           pose_offset, is_ext_fixed);
  cameras_[id] = camera;
}

// Add shot with SE(3) parameter to BundleAdjuster
void BundleAdjuster::add_SE3_shot(const uint32_t id, const Eigen::Matrix<double, 6, 1>& params,
                                  const int flags) {
  const bool is_fixed = flags & FLAG_FIX_PARAMS;
  auto shot           = new Shot(id, params, is_fixed, use_Sim3_cache_);
  shots_[id]          = shot;
}

// Get SE(3) parameter of a shot with specified id
Eigen::Matrix<double, 6, 1> BundleAdjuster::get_SE3_shot_params(const uint32_t id) const {
  const auto shot = shots_.at(id);
  assert(shot->type == Shot::Type::SE3
         && "Requested to get SE3 pose but it is not SE3 shot like Sim3");
  return Eigen::Map<Eigen::Matrix<double, 6, 1>>(shot->params);
}

void BundleAdjuster::add_Sim3_shot(const uint32_t id, const Eigen::Matrix<double, 7, 1>& params,
                                   const int flags) {
  const bool is_fixed = flags & FLAG_FIX_PARAMS;
  auto shot           = new Shot(id, params, is_fixed, use_Sim3_cache_);
  shots_[id]          = shot;
}

// Get Sim(3) parameter of a shot with specified id
Eigen::Matrix<double, 7, 1> BundleAdjuster::get_Sim3_shot_params(const uint32_t id) const {
  const auto shot = shots_.at(id);
  assert(shot->type == Shot::Type::Sim3
         && "Requested to get Sim3 pose but it is not Sim3 shot like SE3");
  return Eigen::Map<Eigen::Matrix<double, 7, 1>>(shot->params);
}

void BundleAdjuster::fix_shot(const uint32_t id) {
  shots_.at(id)->is_fixed = true;
}

void BundleAdjuster::unfix_shot(const uint32_t id) {
  shots_.at(id)->is_fixed = false;
}

// Add point in 3D space to BundleAdjuster
void BundleAdjuster::add_landmark(
    const uint32_t id, const Eigen::Vector3d& coords, const int flags,
    const std::optional<uint32_t> local_coord_shot_id) {
  const bool is_fixed = flags & FLAG_FIX_PARAMS;
  auto landmark       = new Landmark(id, coords, is_fixed);
  if (local_coord_shot_id) {
    landmark->local_coord_system = shots_.at(local_coord_shot_id.value());
  }
  landmarks_[id] = landmark;
}

void BundleAdjuster::remove_landmark(const uint32_t id) {
  auto landmark         = landmarks_.at(id);
  const auto& residuals = landmark->residuals;
  for (auto& residual : residuals) {
    cameras_.at(residual->camera->id)->residuals.erase(residual);
    shots_.at(residual->shot->id)->residuals.erase(residual);
    reprojection_errors_.erase(residual);
    delete residual;
  }
  landmark->residuals.clear();
  landmarks_.erase(id);
}

// Get point coordinates with specified id
Eigen::Vector3d BundleAdjuster::get_landmark_position(const uint32_t id) const {
  const auto landmark = landmarks_.at(id);
  assert(landmark->type == Landmark::Type::Point3D && "Something wrong in getting landmark position (other than 3 is stored)");
  Eigen::Vector3d params;
  for (uint32_t i = 0; i < 3; i++) {
    params(i) = landmark->params[i];
  }
  return params;
}

// Add reprojection error for monocular camera
void BundleAdjuster::add_reprojection_error(
    const uint32_t camera_id, const uint32_t shot_id,
    const uint32_t landmark_id, const Eigen::Vector2d& coords, const double info_val,
    const LossType loss_type, const double loss_scale) {
  if (!cameras_.contains(camera_id)) {
    std::cerr << "Non-registered camera has been referred in creating "
                 "reprojection error"
              << std::endl;
  }
  if (!shots_.contains(shot_id)) {
    std::cerr << "Non-registered shot has been referred in creating "
                 "reprojection error"
              << std::endl;
  }
  if (!landmarks_.contains(landmark_id)) {
    std::cerr << "Non-registered landmark has been referred in creating "
                 "reprojection error"
              << std::endl;
  }
  auto error = new ReprojectionError(
      cameras_.at(camera_id), shots_.at(shot_id), landmarks_.at(landmark_id),
      coords, {info_val, info_val}, ObservationType::MONOCULAR, loss_type,
      loss_scale);
  cameras_.at(camera_id)->residuals.insert(error);
  shots_.at(shot_id)->residuals.insert(error);
  landmarks_.at(landmark_id)->residuals.insert(error);
  reprojection_errors_.insert(error);
}

// Add reprojection error for stereo camera
void BundleAdjuster::add_stereo_reprojection_error(
    const uint32_t camera_id, const uint32_t shot_id,
    const uint32_t landmark_id, const Eigen::Vector3d& coords, const double info_val,
    const LossType loss_type, const double loss_scale) {
  if (!cameras_.contains(camera_id)) {
    std::cerr << "Non-registered camera has been referred in creating "
                 "reprojection error"
              << std::endl;
  }
  if (!shots_.contains(shot_id)) {
    std::cerr << "Non-registered shot has been referred in creating "
                 "reprojection error"
              << std::endl;
  }
  if (!landmarks_.contains(landmark_id)) {
    std::cerr << "Non-registered landmark has been referred in creating "
                 "reprojection error"
              << std::endl;
  }
  auto error = new ReprojectionError(
      cameras_.at(camera_id), shots_.at(shot_id), landmarks_.at(landmark_id),
      coords, {info_val, info_val, info_val}, ObservationType::STEREO,
      loss_type, loss_scale);
  cameras_.at(camera_id)->residuals.insert(error);
  shots_.at(shot_id)->residuals.insert(error);
  landmarks_.at(landmark_id)->residuals.insert(error);
  reprojection_errors_.insert(error);
}

// Add reprojection error for RGB-D camera
void BundleAdjuster::add_depth_reprojection_error(
    const uint32_t camera_id, const uint32_t shot_id,
    const uint32_t landmark_id, const Eigen::Vector3d& coords, const double pt_info_val,
    const double depth_info_val, const LossType loss_type,
    const double loss_scale) {
  assert(cameras_.contains(camera_id) && shots_.contains(shot_id) && landmarks_.contains(landmark_id)
           && "Non-registered element has been referred in creating reprojection error");
  auto error = new ReprojectionError(
      cameras_[camera_id], shots_[shot_id], landmarks_[landmark_id], coords,
      {pt_info_val, pt_info_val, depth_info_val}, ObservationType::DEPTH,
      loss_type, loss_scale);
  cameras_[camera_id]->residuals.insert(error);
  shots_[shot_id]->residuals.insert(error);
  landmarks_[landmark_id]->residuals.insert(error);
  reprojection_errors_.insert(error);
}

// Remove reprojection error from BundleAdjuster
bool BundleAdjuster::remove_reprojection_error(
    ReprojectionError* reprojection_error) {
  const auto res_id = reprojection_error->residual_block_id;
  if (!res_id) {
    return false;
  }
  cameras_.at(reprojection_error->camera->id)
      ->residuals.erase(reprojection_error);
  shots_.at(reprojection_error->shot->id)->residuals.erase(reprojection_error);
  landmarks_.at(reprojection_error->landmark->id)
      ->residuals.erase(reprojection_error);
  if (problem_) {
    problem_->RemoveResidualBlock(res_id);
  }
  reprojection_error->residual_block_id = nullptr;
  return true;
}

// Construct BundleAdjuster problem
void BundleAdjuster::construct_problem() {
  problem_ = std::make_unique<ceres::Problem>();
  for (auto& i : cameras_) {
    auto camera = i.second;
    problem_->AddParameterBlock(
        camera->int_params, int(Camera::NUM_INTRINSIC_PARAMS[camera->model]));
    if (camera->is_int_fixed) {
      problem_->SetParameterBlockConstant(camera->int_params);
    }
    if (camera->has_ext_params) {
      problem_->AddParameterBlock(
          camera->ext_params,
          int(Camera::NUM_EXTRINSIC_PARAMS[camera->ext_type]));
      if (camera->is_ext_fixed) {
        problem_->SetParameterBlockConstant(camera->ext_params);
      }
    }
  }

  for (auto& i : shots_) {
    auto shot = i.second;
    problem_->AddParameterBlock(shot->params,
                                int(Shot::NUM_PARAMS[shot->type]));
    if (shot->is_fixed) {
      problem_->SetParameterBlockConstant(shot->params);
    }
  }

  for (auto& i : landmarks_) {
    auto lm = i.second;
    problem_->AddParameterBlock(lm->params,
                                int(Landmark::NUM_PARAMS[lm->type]));
    if (lm->is_fixed) {
      problem_->SetParameterBlockConstant(lm->params);
    }
  }

  for (auto residual : reprojection_errors_) {
    const auto loss_function
        = CreateLossFunction(residual->loss_type, residual->loss_scale);

    auto cost_function = residual->generate_cost_function(use_analytic_diff_);
    auto parameter_blocks = residual->stack_parameter_blocks();
    auto res_id = problem_->AddResidualBlock(cost_function, loss_function,
                                             parameter_blocks);
    residual->residual_block_id = res_id;
  }
}

// Solve LM method
void BundleAdjuster::fit(uint32_t iterations, const bool dump_progress,
                         const bool dump_full_progress) {
  assert(problem_ && "construct_problem has not be called");

  if (use_Sim3_cache_) {
    enable_Sim3_cache();
  }

  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.linear_solver_type           = ceres::SPARSE_SCHUR;
  options.num_threads                  = int(num_threads_);
  options.max_num_iterations           = int(iterations);
  options.minimizer_progress_to_stdout = dump_progress;
  auto cb = std::make_unique<BAIterationCallback>(*this);
  options.callbacks.push_back(cb.get());

  ceres::Solve(options, problem_.get(), &summary);
  if (dump_full_progress) std::cout << summary.FullReport() << std::endl;

  if (use_Sim3_cache_) {
    disable_Sim3_cache();
  }
}

void BundleAdjuster::clear_Sim3_cache() {
  for (const auto& [shot_id, shot] : shots_) {
    if (shot->cache) {
      shot->cache->clear();
    }
  }
}

void BundleAdjuster::enable_Sim3_cache() {
  clear_Sim3_cache();
  for (const auto& [shot_id, shot] : shots_) {
    if (shot->cache) {
      shot->cache->set_enable(true);
    }
  }
}

void BundleAdjuster::disable_Sim3_cache() {
  for (const auto& [shot_id, shot] : shots_) {
    if (shot->cache) {
      shot->cache->set_enable(false);
    }
  }
}

void BundleAdjuster::break_optimization() { break_required_ = true; }

// Estimate covariance of landmark
// Reference: p.141 Backward propagation of covariance from
//            Andrew, Alex M. "Multiple view geometry in computer vision."
//            Kybernetes (2001).
std::optional<Eigen::Matrix3d> BundleAdjuster::estimate_landmark_covariance(
    const uint32_t landmark_id) const {
  if (!landmarks_.contains(landmark_id)) {
    return std::nullopt;
  }
  const auto landmark  = landmarks_.at(landmark_id);
  const auto residuals = landmark->residuals;

  // TODO : Deal with stereo and depth observation
  // Computing covariance of landmark requires least 2 observations
  if (residuals.size() < 2) {
    return std::nullopt;
  }

  // Information matrix (inverse of covariance matrix) of point coordinates
  Eigen::Matrix3d Omega = Eigen::Matrix3d::Zero();

  for (const auto residual : residuals) {
    // Compute Jacobian from landmark coordinates to image coordinates
    double Jx_array[Landmark::MAX_NUM_PARAMS * residual->n_dim];
    {
      auto jacobians = new double*[residual->num_parameter_blocks_];
      for (size_t i = 0; i < residual->num_parameter_blocks_; i++) {
        jacobians[i] = nullptr;
      }
      jacobians[residual->landmark_index_] = Jx_array;

      const auto cost_function
          = residual->generate_cost_function(use_analytic_diff_);
      const auto parameter_blocks = residual->stack_parameter_blocks();
      const auto loss_function
          = CreateLossFunction(residual->loss_type, residual->loss_scale);
      compute_cost(cost_function, loss_function, parameter_blocks,
                   residual->n_dim, jacobians);
      delete cost_function;
      delete[] jacobians;
    }

    // Rearrange Jacobian as matrix
    Eigen::Matrix<double, 2, 3> Jx;
    Jx << Jx_array[0], Jx_array[1], Jx_array[2], Jx_array[3], Jx_array[4],
        Jx_array[5];

    // TODO : Use 3x3 matrix for stereo observation
    // Information matrix of 2D image opint
    Eigen::Matrix2d Omg_x = Eigen::Matrix2d::Identity();
    Omg_x(0, 0)   = residual->obs_info[0] * residual->obs_info[0];
    Omg_x(1, 1)   = residual->obs_info[1] * residual->obs_info[1];

    // Summed up to get overall infomation matrix
    const Eigen::Matrix3d Omega_part = Jx.transpose() * Omg_x * Jx;
    Omega += Omega_part;
  }
  // Check the singularity of the computation result
  Eigen::Matrix3d Sigma = Omega.fullPivLu().inverse();
  if (Sigma.squaredNorm() < 1e-12
      || std::abs(Sigma.determinant()) / Sigma.squaredNorm() < 1e-12) {
    return std::nullopt;
  }
  return std::move(Sigma);
}

// Reject outlier residuals by checking all residuals
uint32_t BundleAdjuster::reject_outlier_residuals(
    std::vector<ResidualReport>* removed_outlier_indices) {
  uint32_t num_inlier = 0;

  for (auto itr = reprojection_errors_.begin();
       itr != reprojection_errors_.end();) {
    // Compute residual and conduct Chi squared test
    ReprojectionError* residual = *itr;
    const auto cost_function
        = residual->generate_cost_function(use_analytic_diff_);
    const auto parameter_block = residual->stack_parameter_blocks();
    const auto loss_function
        = CreateLossFunction(residual->loss_type, residual->loss_scale);
    const auto& [cost, is_inlier] = compute_cost(
        cost_function, loss_function, parameter_block, residual->n_dim);
    delete loss_function;
    delete cost_function;

    // Remove outlier residual from BundleAdjuster
    if (!is_inlier) {
      remove_reprojection_error(residual);
      if (removed_outlier_indices != nullptr) {
        const uint32_t camera_id   = residual->camera->id;
        const uint32_t shot_id     = residual->shot->id;
        const uint32_t landmark_id = residual->landmark->id;
        // Add id of related elements to report
        removed_outlier_indices->emplace_back(camera_id, shot_id, landmark_id,
                                              cost);
      }
      delete residual;
      itr = reprojection_errors_.erase(itr);
    }
    else {
      num_inlier++;
      itr++;
    }
  }
  return num_inlier;
}

// Reject outlier landmarks by checking residuals for each landmark
void BundleAdjuster::reject_outlier_landmarks(
    std::vector<ResidualReport>* removed_outlier_indices) {
  std::unordered_map<uint32_t,
                     std::set<std::tuple<ReprojectionError*, double, bool>>>
      lm_to_residuals;
  std::unordered_set<uint32_t> outlier_lms;

  for (const auto residual : reprojection_errors_) {
    const uint32_t landmark_id = residual->landmark->id;

    const auto cost_function
        = residual->generate_cost_function(use_analytic_diff_);
    const auto parameter_block = residual->stack_parameter_blocks();
    const auto loss_function
        = CreateLossFunction(residual->loss_type, residual->loss_scale);
    const auto& [cost, is_inlier] = compute_cost(
        cost_function, loss_function, parameter_block, residual->n_dim);
    delete loss_function;
    delete cost_function;

    lm_to_residuals[landmark_id].emplace(residual, cost, is_inlier);

    const bool is_fixed
        = residual->shot->is_fixed || residual->landmark->is_fixed;

    if (!is_inlier && !is_fixed) {
      outlier_lms.insert(landmark_id);
    }
  }

  // Remove outlier landmark and its residuals
  for (const auto& lm_id : outlier_lms) {
    uint32_t num_inlier = 0;
    for (const auto& [_r, _c, is_inlier] : lm_to_residuals[lm_id]) {
      if (is_inlier) {
        num_inlier++;
      }
    }

    bool no_triangle = num_inlier < 2;

    for (const auto& [residual, cost, is_inlier] : lm_to_residuals[lm_id]) {
      // Skip removal if the observation is inlier when there are 2 or more
      // observation after removal
      if (is_inlier && !no_triangle) {
        continue;
      }
      remove_reprojection_error(residual);
      if (removed_outlier_indices != nullptr) {
        const uint32_t camera_id   = residual->camera->id;
        const uint32_t shot_id     = residual->shot->id;
        const uint32_t landmark_id = residual->landmark->id;
        removed_outlier_indices->emplace_back(camera_id, shot_id, landmark_id,
                                              cost);
      }
      reprojection_errors_.erase(residual);
      delete residual;
    }
  }
}

std::map<int, int> BundleAdjuster::evaluate_error(
    const int num_bin_inlier) const {
  std::map<int, int> histogram;
  double sum = 0;
  for (const auto residual : reprojection_errors_) {
    const auto cost_function
        = residual->generate_cost_function(use_analytic_diff_);
    const auto parameter_block = residual->stack_parameter_blocks();
    const auto loss_function
        = CreateLossFunction(residual->loss_type, residual->loss_scale);
    const auto& [cost, is_inlier] = compute_cost(
        cost_function, loss_function, parameter_block, residual->n_dim);
    delete loss_function;
    delete cost_function;

    const int bin = int(cost * 2 / residual->loss_scale * num_bin_inlier);
    histogram[std::min(bin, 2 * num_bin_inlier)]++;
    sum += cost;
  }
  std::cout << "mean: " << sum / double(reprojection_errors_.size())
            << std::endl;
  return histogram;
}

std::string BundleAdjuster::profile_landmark(const uint32_t landmark_id) const {
  const auto landmark     = landmarks_.at(landmark_id);
  const auto residual_set = landmark->residuals;

  std::stringstream ss;

  Eigen::Vector3d grad;
  std::vector<ReprojectionError*> residuals(residual_set.begin(),
                                            residual_set.end());
  std::sort(residuals.begin(), residuals.end());

  Eigen::MatrixXd e(residuals.size() * 2, 1);
  Eigen::MatrixXd J(residuals.size() * 2, 3);
  int i = 0;

  for (const auto& residual : residuals) {
    const auto camera = residual->camera;
    const auto shot   = residual->shot;

    // Input parameters
    const auto parameter_block = residual->stack_parameter_blocks();
    double** params            = new double*[parameter_block.size()];
    // Results
    double res[residual->n_dim];
    // Compute Jacobian from landmark coordinates to image coordinates
    double Jx_array[Landmark::NUM_PARAMS[landmark->type] * residual->n_dim];
    {
      double* jacobians[3] = {nullptr, nullptr, Jx_array};

      const auto cost_function
          = residual->generate_cost_function(use_analytic_diff_);
      cost_function->Evaluate(params, res, jacobians);
      delete cost_function;
    }

    Eigen::Vector2d error;
    error << res[0], res[1];
    Eigen::Matrix<double, 2, 3> Jx;
    Jx << Jx_array[0], Jx_array[1], Jx_array[2], Jx_array[3], Jx_array[4],
        Jx_array[5];

    e.block<2, 1>(i * 2, 0) = error;
    J.block<2, 3>(i * 2, 0) = Jx;

    double error_sq = error.squaredNorm();

    double pos_l[3];
    double dummy1[3][7];
    double dummy2[3][3];
    if (shot->type == Shot::Type::SE3) {
      lie::SE3::map(shot->params, landmark->params, pos_l, false, dummy1, false,
                    dummy2);
    }
    else if (shot->type == Shot::Type::Sim3) {
      lie::Sim3::map(shot->params, landmark->params, pos_l, false, dummy1,
                     false, dummy2);
    }
    double rep_pt[2];
    bool front = reproject(camera->int_params, camera->model, pos_l, rep_pt,
                           residual->obs_type, false, false, nullptr, nullptr);

    ss << "cam: " << camera->id << ", sht: " << shot->id << ", ";
    ss << "obs: (" << residual->obs[0] << ", " << residual->obs[1] << "), ";
    ss << "rep: (" << rep_pt[0] << ", " << rep_pt[1] << ", "
       << (front ? "front" : "back") << "), ";
    ss << "err_sq: " << error_sq << std::endl;

    grad += Jx.transpose() * error;

    i++;
  }

  Eigen::Matrix3d H    = J.transpose() * J;
  Eigen::Vector3d delta = -H.inverse() * J.transpose() * e;

  ss << "gradient: " << grad.transpose();
  ss << " delta_x: " << delta.transpose();

  return ss.str();
}
} // namespace bundlefit
