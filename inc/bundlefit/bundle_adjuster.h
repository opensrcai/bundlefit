#ifndef BUNDLEFIT_BUNDLE_ADJUSTER_H
#define BUNDLEFIT_BUNDLE_ADJUSTER_H

#include <atomic>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "bundlefit/types.h"

namespace ceres {
class Problem;
} // namespace ceres

namespace bundlefit {

struct Camera;
struct Shot;
struct Landmark;
struct ReprojectionError;

struct ResidualReport {
  ResidualReport(const uint32_t _camera_id, const uint32_t _shot_id,
                 const uint32_t _landmark_id, const double _cost)
      : camera_id(_camera_id)
      , shot_id(_shot_id)
      , landmark_id(_landmark_id)
      , cost(_cost) {}
  uint32_t camera_id;
  uint32_t shot_id;
  uint32_t landmark_id;
  double cost;
};

const int FLAG_OPTIMIZE_PARAMS      = 0b0000;
const int FLAG_FIX_INTRINSIC_PARAMS = 0b0001;
const int FLAG_FIX_EXTRINSIC_PARAMS = 0b0010;
const int FLAG_FIX_PARAMS           = 0b0001;

enum class LossType {
  TRIVIAL = 0,
  HUBER_LOSS,
  SOFT_L1,
  CAUCHY
};

class BundleAdjuster {
  friend class BAIterationCallback;

public:
  BundleAdjuster(const uint32_t num_threads_  = 2,
                 const bool use_analytic_diff = true,
                 const bool use_Sim3_cache    = true);
  ~BundleAdjuster();

  void add_perspective_camera(const uint32_t id, const double fx,
                              const double fy, const double cx, const double cy,
                              const int flags);
  void add_perspective_camera(const uint32_t id, const double fx,
                              const double fy, const double cx, const double cy,
                              double pixel_baseline, const int flags);
  void add_perspective_camera(const uint32_t id, const double fx,
                              const double fy, const double cx, const double cy,
                              const Eigen::Matrix<double, 6, 1>& pose_offset, const int flags);
  void add_equirectangular_camera(const uint32_t id, uint32_t rows,
                                  uint32_t cols, const int flags);
  void add_equirectangular_camera(const uint32_t id, uint32_t rows,
                                  uint32_t cols, const Eigen::Matrix<double, 6, 1>& pose_offset,
                                  const int flags);

  void add_SE3_shot(const uint32_t id, const Eigen::Matrix<double, 6, 1>& params, const int);
  Eigen::Matrix<double, 6, 1> get_SE3_shot_params(const uint32_t id) const;

  void add_Sim3_shot(const uint32_t id, const Eigen::Matrix<double, 7, 1>& params, const int);
  Eigen::Matrix<double, 7, 1> get_Sim3_shot_params(const uint32_t id) const;

  void fix_shot(const uint32_t id);
  void unfix_shot(const uint32_t id);

  void add_landmark(const uint32_t id, const Eigen::Vector3d& position, const int flags,
                    const std::optional<uint32_t> local_coord_shot_id
                    = std::nullopt);
  void remove_landmark(const uint32_t id);
  Eigen::Vector3d get_landmark_position(const uint32_t id) const;

  void add_reprojection_error(const uint32_t camera_id, const uint32_t shot_id,
                              const uint32_t landmark_id, const Eigen::Vector2d& obs,
                              const double info_val, const LossType loss_type,
                              const double loss_scale);

  void add_stereo_reprojection_error(const uint32_t camera_id,
                                     const uint32_t shot_id,
                                     const uint32_t landmark_id,
                                     const Eigen::Vector3d& obs, const double info_val,
                                     const LossType loss_type,
                                     const double loss_scale);

  void add_depth_reprojection_error(const uint32_t camera_id,
                                    const uint32_t shot_id,
                                    const uint32_t landmark_id,
                                    const Eigen::Vector3d& obs, const double pt_info_val,
                                    const double depth_info_val,
                                    const LossType loss_type,
                                    const double loss_scale);

  void construct_problem();
  void fit(const uint32_t iterations, const bool dump_progress = false,
           const bool dump_full_progress = false);

  void break_optimization();

  std::optional<Eigen::Matrix3d> estimate_landmark_covariance(
      const uint32_t landmark_id) const;

  //! Remove error terms with reprojection error above the threshold
  uint32_t reject_outlier_residuals(std::vector<ResidualReport>* outlier_indices
                                    = nullptr);

  void reject_outlier_landmarks(std::vector<ResidualReport>* outlier_indices
                                = nullptr);

  std::map<int, int> evaluate_error(const int bins_inlier = 10) const;

  std::string profile_landmark(const uint32_t landmark_id) const;

protected:
  std::unique_ptr<ceres::Problem> problem_;
  const uint32_t num_threads_;
  const bool use_analytic_diff_;
  const bool use_Sim3_cache_;

  std::atomic<bool> break_required_{false};

  std::unordered_map<uint32_t, Camera*> cameras_;
  std::unordered_map<uint32_t, Shot*> shots_;
  std::unordered_map<uint32_t, Landmark*> landmarks_;

  std::unordered_set<ReprojectionError*> reprojection_errors_;

  void clear_Sim3_cache();
  void enable_Sim3_cache();
  void disable_Sim3_cache();

  bool remove_reprojection_error(ReprojectionError* reprojection_error);
};

} // namespace bundlefit

#endif // BUNDLEFIT_BUNDLE_ADJUSTER_H
