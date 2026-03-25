#include "bundle_adjuster.h"

#include <assert.h>
#include <ceres/problem.h>
#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "bundlefit/bundle_adjuster.h"
#include "bundlefit/types.h"
#include "nanobind/eigen/dense.h"
#include "nanobind/stl/map.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/vector.h"

namespace bundlefit {

class BundleAdjusterPython : public BundleAdjuster {
public:
  BundleAdjusterPython(const uint32_t num_threads_     = 2,
                       const bool use_analytic_diff    = true,
                       const bool use_dynamic_autodiff = true) {
    BundleAdjuster(num_threads_, use_analytic_diff, use_dynamic_autodiff);
  }

  ~BundleAdjusterPython() {}

  void add_perspective_camera(const uint32_t id, const double fx,
                              const double fy, const double cx, const double cy,
                              const int flags) {
    BundleAdjuster::add_perspective_camera(id, fx, fy, cx, cy, flags);
  }
  void add_perspective_cameras(const std::vector<uint32_t>& id,
                               const std::vector<double>& fx,
                               const std::vector<double>& fy,
                               const std::vector<double>& cx,
                               const std::vector<double>& cy,
                               const std::vector<int>& flags) {
    const size_t n = std::min(
        {id.size(), fx.size(), fy.size(), cx.size(), cy.size(), flags.size()});
    for (size_t i = 0; i < n; ++i) {
      this->add_perspective_camera(id[i], fx[i], fy[i], cx[i], cy[i], flags[i]);
    }
  }

  void add_equirectangular_camera(const uint32_t id, uint32_t rows,
                                  uint32_t cols, const bool flags = true) {
    BundleAdjuster::add_equirectangular_camera(id, rows, cols, flags);
  }
  void add_equirectangular_cameras(const std::vector<uint32_t>& id,
                                   const std::vector<uint32_t>& rows,
                                   const std::vector<uint32_t>& cols,
                                   const std::vector<int>& flags) {
    const size_t n
        = std::min({id.size(), rows.size(), cols.size(), flags.size()});
    for (size_t i = 0; i < n; ++i) {
      this->add_equirectangular_camera(id[i], rows[i], cols[i], flags[i]);
    }
  }

  void add_SE3_shot(const uint32_t id, const Eigen::Matrix<double, 6, 1>& params, const int flags) {
    BundleAdjuster::add_SE3_shot(id, params, flags);
  }
  void add_SE3_shots(
      const std::vector<uint32_t>& id,
      const Eigen::Matrix<double, -1, 6, Eigen::RowMajorBit>& params,
      const std::vector<int>& flags) {
    const size_t n = std::min(
        {id.size(), static_cast<size_t>(params.rows()), flags.size()});
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Matrix<double, 6, 1> param = params.transpose().col(i);
      this->add_SE3_shot(id[i], param, flags[i]);
    }
  }

  Eigen::Matrix<double, 6, 1> get_SE3_shot_params(const uint32_t id) const {
    return BundleAdjuster::get_SE3_shot_params(id);
  }

  Eigen::Matrix<double, -1, 6, Eigen::RowMajorBit> get_SE3_shots_params(
      const std::vector<uint32_t>& ids) const {
    const size_t num_shots = ids.size();
    Eigen::Matrix<double, -1, 6, Eigen::RowMajorBit> ret(
        static_cast<int>(num_shots), 6);

    int i = 0;
    for (const uint32_t id : ids) {
      const Eigen::Matrix<double, 6, 1> se3_vec = BundleAdjuster::get_SE3_shot_params(id);
      for (int j = 0; j < 6; ++j) {
        ret(i, j) = se3_vec[j];
      }
      ++i;
    }

    return ret;
  }

  void add_landmark(const uint32_t id, const Eigen::Vector3d& position, const int flags,
                    const std::optional<uint32_t> local_coord_shot_id
                    = std::nullopt) {
    BundleAdjuster::add_landmark(id, position, flags, local_coord_shot_id);
  }
  void add_landmarks(
      const std::vector<uint32_t>& id,
      const Eigen::Matrix<double, -1, 3, Eigen::RowMajorBit>& coords,
      const std::vector<int>& flags,
      const std::vector<std::optional<uint32_t>>& local_coord_shot_id) {
    const size_t n = std::min({id.size(), static_cast<size_t>(coords.rows()),
                               flags.size(), local_coord_shot_id.size()});
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Vector3d coord = coords.transpose().col(i);
      this->add_landmark(id[i], coord, flags[i], local_coord_shot_id[i]);
    }
  }
  Eigen::Vector3d get_landmark_position(const uint32_t id) const {
    return BundleAdjuster::get_landmark_position(id);
  }

  Eigen::Matrix<double, -1, 3, Eigen::RowMajorBit> get_landmark_positions(
      const std::vector<uint32_t>& ids) const {
    const size_t num_landmarks = ids.size();
    Eigen::Matrix<double, -1, 3, Eigen::RowMajorBit> ret(
        static_cast<int>(num_landmarks), 3);

    int i = 0;
    for (const uint32_t id : ids) {
      const Eigen::Vector3d pos = BundleAdjuster::get_landmark_position(id).transpose();
      for (int j = 0; j < 3; ++j) {
        ret(i, j) = pos[j];
      }
      ++i;
    }

    return ret;
  }

  void add_reprojection_error(const uint32_t camera_id, const uint32_t shot_id,
                              const uint32_t landmark_id, const Eigen::Vector2d& obs,
                              const double info_val, const LossType loss_type,
                              const double loss_scale) {
    BundleAdjuster::add_reprojection_error(camera_id, shot_id, landmark_id, obs,
                                           info_val, loss_type, loss_scale);
  }

  void add_reprojection_errors(
      const std::vector<uint32_t>& camera_id,
      const std::vector<uint32_t>& shot_id,
      const std::vector<uint32_t>& landmark_id,
      const Eigen::Matrix<double, -1, 2, Eigen::RowMajorBit>& obs,
      const std::vector<double>& info_val, const std::vector<int>& loss_type,
      const std::vector<double>& loss_scale) {
    const size_t n
        = std::min({camera_id.size(), shot_id.size(), landmark_id.size(),
                    static_cast<size_t>(obs.rows()), info_val.size()});
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Vector2d coord = obs.transpose().col(i);
      this->add_reprojection_error(
          camera_id[i], shot_id[i], landmark_id[i], coord, info_val[i],
          static_cast<LossType>(loss_type[i]), loss_scale[i]);
    }
  }

  void construct_problem() { BundleAdjuster::construct_problem(); }

  void fit(const uint32_t iterations, const bool dump_progress = false,
           const bool dump_full_progress = false) {
    BundleAdjuster::fit(iterations, dump_progress, dump_full_progress);
  }

  std::optional<Eigen::Matrix3d> estimate_landmark_covariance(
      const uint32_t landmark_id) const {
    return BundleAdjuster::estimate_landmark_covariance(landmark_id);
  }

  uint32_t reject_outlier_residuals(std::vector<ResidualReport>* outlier_indices
                                    = nullptr) {
    return BundleAdjuster::reject_outlier_residuals(outlier_indices);
  }

  void reject_outlier_landmarks(std::vector<ResidualReport>* outlier_indices
                                = nullptr) {
    BundleAdjuster::reject_outlier_landmarks(outlier_indices);
  }

  std::map<int, int> evaluate_error(const int bins_inlier = 10) const {
    return BundleAdjuster::evaluate_error(bins_inlier);
  }
};

void WrapBundleAdjuster(nanobind::module_& module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  module.attr("FLAG_OPTIMIZE_PARAMS") = nanobind::int_(FLAG_OPTIMIZE_PARAMS);
  module.attr("FLAG_FIX_INTRINSIC_PARAMS")
      = nanobind::int_(FLAG_FIX_INTRINSIC_PARAMS);
  module.attr("FLAG_FIX_EXTRINSIC_PARAMS")
      = nanobind::int_(FLAG_FIX_EXTRINSIC_PARAMS);
  module.attr("FLAG_FIX_PARAMS") = nanobind::int_(FLAG_FIX_PARAMS);

  nb::enum_<LossType> loss_type(module, "LossType", nb::is_arithmetic());

  loss_type.value("TRIVIAL", LossType::TRIVIAL)
      .value("HUBER_LOSS", LossType::HUBER_LOSS)
      .value("SOFT_L1", LossType::SOFT_L1)
      .value("CAUCHY", LossType::CAUCHY)
      .export_values();

  nb::class_<BundleAdjusterPython> bundle_adjuster(module, "BundleAdjuster");

  bundle_adjuster.def(nb::init<const uint32_t, const bool, const bool>())
      .def("add_perspective_camera",
           &BundleAdjusterPython::add_perspective_camera, "id"_a, "fx"_a,
           "fy"_a, "cx"_a, "cy"_a, "flags"_a)
      .def("add_perspective_cameras",
           &BundleAdjusterPython::add_perspective_cameras, "id"_a, "fx"_a,
           "fy"_a, "cx"_a, "cy"_a, "flags"_a)
      .def("add_equirectangular_camera",
           &BundleAdjusterPython::add_equirectangular_camera, "id"_a, "rows"_a,
           "cols"_a, "flags "_a = true)
      .def("add_equirectangular_cameras",
           &BundleAdjusterPython::add_equirectangular_cameras, "id"_a, "rows"_a,
           "cols"_a, "flags "_a)
      .def("add_SE3_shot", &BundleAdjusterPython::add_SE3_shot, "id"_a,
           "params"_a, "flags"_a = false)
      .def("add_SE3_shots", &BundleAdjusterPython::add_SE3_shots, "id"_a,
           "params"_a, "flags"_a)
      .def("get_SE3_shot_params", &BundleAdjusterPython::get_SE3_shot_params,
           "id"_a)
      .def("get_SE3_shots_params", &BundleAdjusterPython::get_SE3_shots_params,
           "ids"_a)
      .def("add_landmark", &BundleAdjusterPython::add_landmark, "id"_a,
           "position"_a, "flags"_a, "local_coord_shot_id"_a.none())
      .def("add_landmarks", &BundleAdjusterPython::add_landmarks, "id"_a,
           "coords"_a, "flags"_a, "local_coord_shot_id"_a)
      .def("get_landmark_position",
           &BundleAdjusterPython::get_landmark_position, "id"_a)
      .def("get_landmark_positions",
           &BundleAdjusterPython::get_landmark_positions, "ids"_a)
      .def("add_reprojection_error",
           &BundleAdjusterPython::add_reprojection_error, "camera_id"_a,
           "shot_id"_a, "landmark_id"_a, "obs"_a, "info_val"_a, "loss_type"_a,
           "loss_scale"_a)
      .def("add_reprojection_errors",
           &BundleAdjusterPython::add_reprojection_errors, "camera_id"_a,
           "shot_id"_a, "landmark_id"_a, "obs"_a, "info_val"_a, "loss_type"_a,
           "loss_scale"_a)
      .def("construct_problem", &BundleAdjusterPython::construct_problem)
      .def("fit", &BundleAdjusterPython::fit, "iterations"_a,
           "dump_progress"_a = false, "dump_full_progress"_a = false)
      .def("estimate_landmark_covariance",
           &BundleAdjusterPython::estimate_landmark_covariance)
      .def("reject_outlier_residuals",
           &BundleAdjusterPython::reject_outlier_residuals,
           "outlier_indices"_a = nullptr)
      .def("reject_outlier_landmarks",
           &BundleAdjusterPython::reject_outlier_landmarks,
           "outlier_indices"_a = nullptr)
      .def("evaluate_error", &BundleAdjusterPython::evaluate_error,
           "bins_inlier"_a = 10);
}

} // namespace bundlefit
