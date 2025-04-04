#ifndef BUNDLEFIT_CAMERA_H
#define BUNDLEFIT_CAMERA_H

#include <ceres/jet.h>

#include <unordered_set>

#include "bundlefit/constant.h"
#include "bundlefit/types.h"
#include "bundlefit/utils.h"

namespace bundlefit {
struct ReprojectionError;

// Camera struct for all projection model
struct Camera {
  // Projection models of camera
  enum class ProjectionModel {
    PERSPECTIVE = 0,
    EQUIRECTANGULAR
  };

  static constexpr auto NUM_INTRINSIC_PARAMS
      = make_const_dict<ProjectionModel, uint32_t>(
          {{ProjectionModel::PERSPECTIVE, 5},
           {ProjectionModel::EQUIRECTANGULAR, 2}});
  static constexpr auto MAX_NUM_INT_PARAMS = NUM_INTRINSIC_PARAMS.max();

  enum class ExtrinsicType {
    SE3 = 0,
    Sim3,
    None
  };
  static constexpr auto NUM_EXTRINSIC_PARAMS
      = make_const_dict<ExtrinsicType, uint32_t>({{ExtrinsicType::SE3, 6},
                                                  {ExtrinsicType::Sim3, 7},
                                                  {ExtrinsicType::None, 0}});
  static constexpr auto MAX_NUM_EXT_PARAMS = NUM_INTRINSIC_PARAMS.max();

  template <typename T, size_t Ni, int Ne>
  Camera(const uint32_t _id, const ProjectionModel _model,
         const T (&_int_params)[Ni], const bool _is_int_fixed,
         const Vec_t<Ne>& _ext_params, const bool _is_ext_fixed)
      : id(_id)
      , model(_model)
      , ext_type(Ne > 0 ? ExtrinsicType::SE3 : ExtrinsicType::None)
      , int_params(new double[Camera::NUM_INTRINSIC_PARAMS[model]])
      , is_int_fixed(_is_int_fixed)
      , ext_params(new double[Camera::NUM_EXTRINSIC_PARAMS[ext_type]])
      , is_ext_fixed(_is_ext_fixed)
      , has_ext_params(ext_type != ExtrinsicType::None) {
    for (uint32_t i = 0; i < Ni; i++) {
      int_params[i] = _int_params[i];
    }
    for (uint32_t i = 0; i < Camera::NUM_EXTRINSIC_PARAMS[ext_type]; i++) {
      ext_params[i] = _ext_params(i);
    }
  }
  template <typename T, size_t Ni>
  Camera(const uint32_t _id, const ProjectionModel _model,
         const T (&_int_params)[Ni], const bool _is_int_fixed)
      : Camera(_id, _model, _int_params, _is_int_fixed, Vec_t<0>{}, true) {}

  ~Camera() {
    delete[] int_params;
    delete[] ext_params;
  }

  const uint32_t id;
  const ProjectionModel model;
  const ExtrinsicType ext_type;

  // Intrinsic parameters for camera
  double* int_params;
  const bool is_int_fixed;

  // The camera's pose in shot local coordinate.
  // For a normal camera, no need to provide.
  // If this camera is one of multi-camera system,
  // SE3 pose parameters(local coords -> camera coords) should be set.
  double* ext_params;
  const bool is_ext_fixed;
  const bool has_ext_params;

  // Reprojection errors related to this camera
  std::unordered_set<ReprojectionError*> residuals;
};

// Perspective camera derivation
struct PerspectiveCamera : public Camera {
  // Vanilla pinhole camera
  PerspectiveCamera(const uint32_t _id, const double fx, const double fy,
                    const double cx, const double cy, const bool _is_int_fixed)
      : Camera(_id, ProjectionModel::PERSPECTIVE, {fx, fy, cx, cy},
               _is_int_fixed) {}
  PerspectiveCamera(const uint32_t _id, const double fx, const double fy,
                    const double cx, const double cy,
                    const double pixel_baseline, const bool _is_int_fixed)
      : Camera(_id, ProjectionModel::PERSPECTIVE,
               {fx, fy, cx, cy, pixel_baseline}, _is_int_fixed) {}
  PerspectiveCamera(const uint32_t _id, const double fx, const double fy,
                    const double cx, const double cy, const bool _is_int_fixed,
                    const Eigen::Matrix<double, 6, 1>& _pose_offset,
                    const bool _is_ext_fixed)
      : Camera(_id, ProjectionModel::PERSPECTIVE, {fx, fy, cx, cy},
               _is_int_fixed, _pose_offset, _is_ext_fixed) {
    assert(
        _is_ext_fixed
        && "Optimization for external params in camera is not implemented "
           "yet.");
  }

  template <typename T>
  static T fx(const T* _params) {
    return T(_params[0]);
  }
  template <typename T>
  static T fy(const T* _params) {
    return T(_params[1]);
  }
  template <typename T>
  static T cx(const T* _params) {
    return T(_params[2]);
  }
  template <typename T>
  static T cy(const T* _params) {
    return T(_params[3]);
  }
  template <typename T>
  static T pixel_baseline(const T* _params) {
    return T(_params[4]);
  }

  template <typename T>
  static inline bool reproject(const T* camera_params, const T* pos_l, T* pos_i,
                               const ObservationType obs_type) {
    pos_i[0] = PerspectiveCamera::fx(camera_params) * pos_l[0] / pos_l[2]
               + PerspectiveCamera::cx(camera_params);
    pos_i[1] = PerspectiveCamera::fy(camera_params) * pos_l[1] / pos_l[2]
               + PerspectiveCamera::cy(camera_params);
    if (obs_type == ObservationType::STEREO) {
      pos_i[2] = pos_i[0]
                 - PerspectiveCamera::pixel_baseline(camera_params) / pos_l[2];
    }
    else if (obs_type == ObservationType::DEPTH) {
      pos_i[2] = pos_l[2];
    }
    return pos_l[2] > T(0);
  }

  static inline bool reproject(const double* camera_params, const double* pos_l,
                               double* pos_i, const ObservationType obs_type,
                               const bool derive_camera,
                               const bool derive_point, double di_dc[3][5],
                               double di_dpl[3][3]) {
    const double inv_z = 1.0 / pos_l[2];
    pos_i[0]           = PerspectiveCamera::fx(camera_params) * pos_l[0] * inv_z
               + PerspectiveCamera::cx(camera_params);
    pos_i[1] = PerspectiveCamera::fy(camera_params) * pos_l[1] * inv_z
               + PerspectiveCamera::cy(camera_params);
    if (obs_type == ObservationType::STEREO) {
      pos_i[2]
          = pos_i[0] - PerspectiveCamera::pixel_baseline(camera_params) * inv_z;
    }
    else if (obs_type == ObservationType::DEPTH) {
      pos_i[2] = pos_l[2];
    }

    if (derive_point) {
      const double du_dx = PerspectiveCamera::fx(camera_params) * inv_z;
      const double dv_dy = PerspectiveCamera::fy(camera_params) * inv_z;
      di_dpl[0][0]       = du_dx;
      di_dpl[0][1]       = 0.0;
      di_dpl[1][0]       = 0.0;
      di_dpl[1][1]       = dv_dy;
      di_dpl[0][2]       = -du_dx * pos_l[0] * inv_z;
      di_dpl[1][2]       = -dv_dy * pos_l[1] * inv_z;

      if (obs_type == ObservationType::STEREO) {
        di_dpl[2][0] = du_dx;
        di_dpl[2][1] = 0;
        di_dpl[2][2] = di_dpl[0][2]
                       + PerspectiveCamera::pixel_baseline(camera_params)
                             * inv_z * inv_z;
      }
      else if (obs_type == ObservationType::DEPTH) {
        di_dpl[2][0] = 0;
        di_dpl[2][1] = 0;
        di_dpl[2][2] = 1;
      }
    }

    if (derive_camera) {
      di_dc[0][0] = pos_l[0] * inv_z;
      di_dc[0][1] = 0;
      di_dc[0][2] = 1;
      di_dc[0][3] = 0;
      di_dc[0][4] = 0;
      di_dc[1][0] = 0;
      di_dc[1][1] = pos_l[1] * inv_z;
      di_dc[1][2] = 0;
      di_dc[1][3] = 1;
      di_dc[1][4] = 0;
      if (obs_type == ObservationType::STEREO) {
        di_dc[2][0] = pos_l[0] * inv_z;
        di_dc[2][1] = 0;
        di_dc[2][2] = 1;
        di_dc[2][3] = 0;
        di_dc[2][4] = -1 * inv_z;
      }
      else if (obs_type == ObservationType::DEPTH) {
        di_dc[2][0] = 0;
        di_dc[2][1] = 0;
        di_dc[2][2] = 0;
        di_dc[2][3] = 0;
        di_dc[2][4] = 0;
      }
    }
    return pos_l[2] > 0;
  }
};

// Equirectangular camera derivation
struct EquirectangularCamera : public Camera {
  EquirectangularCamera(const uint32_t _id, const uint32_t num_rows,
                        const uint32_t num_cols, const bool _is_fixed)
      : Camera(_id, ProjectionModel::EQUIRECTANGULAR, {num_rows, num_cols},
               _is_fixed) {
    assert(_is_fixed
           && "Intrinsic parameters in equirectangular camera must be fixed.");
  }
  EquirectangularCamera(const uint32_t _id, const uint32_t num_rows,
                        const uint32_t num_cols, const bool _is_int_fixed,
                        const Eigen::Matrix<double, 6, 1>& _pose_offset,
                        const bool _is_ext_fixed)
      : Camera(_id, ProjectionModel::EQUIRECTANGULAR, {num_rows, num_cols},
               _is_int_fixed, _pose_offset, _is_ext_fixed) {
    assert(_is_int_fixed
           && "Intrinsic parameters in equirectangular camera must be fixed.");
  }
  template <typename T>
  static T rows(const T* _params) {
    return T(_params[0]);
  }
  template <typename T>
  static T cols(const T* _params) {
    return T(_params[1]);
  }

  template <typename T>
  static inline bool reproject(
      const T* camera_params, const T* pos_l, T* pos_i,
      [[maybe_unused]] const ObservationType obs_type) {
    assert(
        obs_type == ObservationType::MONOCULAR
        && "Only monocular observation is allowed for equirectangular camera.");
    const double rows
        = utils::convert_to_double(EquirectangularCamera::rows(camera_params));
    const double cols
        = utils::convert_to_double(EquirectangularCamera::cols(camera_params));
    const T theta = ceres::atan2(pos_l[0], pos_l[2]);
    const T phi   = ceres::atan2(
        -pos_l[1], ceres::sqrt(pos_l[0] * pos_l[0] + pos_l[2] * pos_l[2]));
    pos_i[0] = theta * T(0.5 * cols / T(kPi<double>)) + T(0.5 * cols);
    pos_i[1] = -phi * T(rows / T(kPi<double>)) + T(0.5 * rows);
    return true;
  }

  static inline bool reproject(const double* camera_params, const double* pos_l,
                               double* pos_i,
                               [[maybe_unused]] const ObservationType obs_type,
                               const bool derive_camera,
                               const bool derive_point, double di_dc[3][5],
                               double di_dpl[3][3]) {
    assert(
        obs_type == ObservationType::MONOCULAR
        && "Only monocular observation is allowed for equirectangular camera.");
    const auto rows    = EquirectangularCamera::rows(camera_params);
    const auto cols    = EquirectangularCamera::cols(camera_params);
    const double theta = ceres::atan2(pos_l[0], pos_l[2]);
    const double phi   = ceres::atan2(
        -pos_l[1], ceres::sqrt(pos_l[0] * pos_l[0] + pos_l[2] * pos_l[2]));
    pos_i[0] = theta * 0.5 * cols / kPi<double> + 0.5 * cols;
    pos_i[1] = -phi * rows / kPi<double> + 0.5 * rows;

    if (derive_point) {
      const double a1          = 0.5 * cols / kPi<double>;
      const double a2          = rows / kPi<double>;
      const double x_sq_z_sq   = pos_l[0] * pos_l[0] + pos_l[2] * pos_l[2];
      const double sq_norm     = x_sq_z_sq + pos_l[1] * pos_l[1];
      const double phi_inv_den = 1.0 / std::sqrt(x_sq_z_sq) / sq_norm;

      di_dpl[0][0] = a1 * pos_l[2] / x_sq_z_sq;
      di_dpl[0][1] = 0.0;
      di_dpl[0][2] = -a1 * pos_l[0] / x_sq_z_sq;
      di_dpl[1][0] = -a2 * pos_l[0] * pos_l[1] * phi_inv_den;
      di_dpl[1][1] = a2 * x_sq_z_sq * phi_inv_den;
      di_dpl[1][2] = -a2 * pos_l[1] * pos_l[2] * phi_inv_den;
    }

    if (derive_camera) {
      for (uint32_t i = 0; i < NUM_RESIDUALS[obs_type]; i++) {
        for (uint32_t j = 0;
             j < NUM_INTRINSIC_PARAMS[ProjectionModel::EQUIRECTANGULAR]; j++) {
          di_dc[i][j] = 0.0;
        }
      }
    }
    return true;
  }
};

// Reprojection function for auto differentiation
template <typename T>
inline bool reproject(const T* camera_params,
                      const Camera::ProjectionModel model, const T* pos_l,
                      T* pos_i, const ObservationType obs_type) {
  if (model == Camera::ProjectionModel::PERSPECTIVE) {
    return PerspectiveCamera::reproject(camera_params, pos_l, pos_i, obs_type);
  }
  else if (model == Camera::ProjectionModel::EQUIRECTANGULAR) {
    return EquirectangularCamera::reproject(camera_params, pos_l, pos_i,
                                            obs_type);
  }
  else {
    assert(false && "Unknown camera type");
    pos_i[0] = T(0.0);
    pos_i[1] = T(0.0);
    return false;
  }
}

// Reprojection function for analytical differentiation
inline bool reproject(const double* camera_params,
                      const Camera::ProjectionModel model, const double* pos_l,
                      double* pos_i, const ObservationType obs_type,
                      const bool derive_camera, const bool derive_point,
                      double di_dc[3][5], double di_dpl[3][3]) {
  if (model == Camera::ProjectionModel::PERSPECTIVE) {
    return PerspectiveCamera::reproject(camera_params, pos_l, pos_i, obs_type,
                                        derive_camera, derive_point, di_dc,
                                        di_dpl);
  }
  else if (model == Camera::ProjectionModel::EQUIRECTANGULAR) {
    return EquirectangularCamera::reproject(camera_params, pos_l, pos_i,
                                            obs_type, derive_camera,
                                            derive_point, di_dc, di_dpl);
  }
  else {
    assert(false && "Unknown camera type");
    return false;
  }
}
} // namespace bundlefit
#endif // BUNDLEFIT_CAMERA_H
