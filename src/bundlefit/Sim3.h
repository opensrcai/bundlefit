#ifndef BUNDLEFIT_SIM3_H
#define BUNDLEFIT_SIM3_H

#include <ceres/jet.h>
#include <ceres/rotation.h>

#include "bundlefit/types.h"

namespace bundlefit {

namespace pgo_core {

template <typename t>
using ceres_mat_t = t[9];

template <typename t>
using ceres_vec_t = t[3];

// 3x3 matrix - matrix multiplication
template <typename T>
void multi_mat(const ceres_mat_t<T> M1, const ceres_mat_t<T> M2,
               ceres_mat_t<T> MR) {
  for (uint32_t j = 0; j < 3; j++) {
    for (uint32_t i = 0; i < 3; i++) {
      MR[i + 3 * j] = M1[i] * M2[3 * j] + M1[i + 3] * M2[3 * j + 1]
                      + M1[i + 6] * M2[3 * j + 2];
    }
  }
}

// 3x3 matrix - 3d vector multipliation
template <typename T>
void multi_vec(const ceres_mat_t<T> M1, const ceres_vec_t<T> M2,
               ceres_vec_t<T> MR) {
  for (uint32_t i = 0; i < 3; i++) {
    MR[i] = M1[i] * M2[0] + M1[i + 3] * M2[1] + M1[i + 6] * M2[2];
  }
}

// Determinant of 3x3 matrix
template <typename T>
T determinant(const ceres_mat_t<T> M) {
  return M[0] * (M[4] * M[8] - M[5] * M[7]) + M[3] * (M[7] * M[2] - M[8] * M[1])
         + M[6] * (M[1] * M[5] - M[2] * M[4]);
}

// TODO bugs ?
// Multiply inverse of M and vector V
template <typename T>
void multi_mat_inv(const ceres_mat_t<T> M, const ceres_mat_t<T> V,
                   ceres_mat_t<T> VR) {
  T den = determinant(M);
  for (uint32_t i = 0; i < 3; i++) {
    T M_num[9];
    for (unsigned jy = 0; jy < 3; jy++) {
      for (unsigned jx = 0; jx < 3; jx++) {
        if (jx == i) {
          M_num[jy + jx * 3] = V[jy];
        }
        else {
          M_num[jy + jx * 3] = M[jy + jx * 3];
        }
      }
    }
    T num = determinant(M_num);
    VR[i] = num / den;
  }
}

// Compute exponential map from sim(3) to Sim(3)
template <typename T>
void exp_Sim3(const T params[7], T rot[9], T trans[3], T* scale) {
  ceres::AngleAxisToRotationMatrix(params, rot);
  *scale = ceres::exp(params[6]);

  for (uint32_t i = 0; i < 3; i++) {
    trans[i] = params[i + 3];
  }
}

// Compute logarithm map from Sim(3) to sim(3)
template <typename T>
void log_Sim3(const T rot[9], const T trans[3], const T scale, T params[7]) {
  T rvec[3];
  ceres::RotationMatrixToAngleAxis(rot, rvec);
  for (uint32_t i = 0; i < 3; i++) {
    params[i] = rvec[i];
  }
  params[6] = ceres::log(scale);

  for (uint32_t i = 0; i < 3; i++) {
    params[i + 3] = trans[i];
  }
}

// Sim(3) struct
template <typename T>
struct Sim3_t {
  T params[7];
  explicit Sim3_t(const T _params[7]) {
    for (uint32_t i = 0; i < 7; i++) {
      params[i] = _params[i];
    }
  }
  Sim3_t(const T R[9], const T t[3], const T scale) {
    log_Sim3(R, t, scale, params);
  }
  template <typename C>
  Sim3_t(const C R_c[9], const C t_c[3], const C scale_c) {
    T R[9];
    T t[3];
    T scale;
    for (uint32_t i = 0; i < 9; i++) {
      R[i] = T(R_c[i]);
    }
    for (uint32_t i = 0; i < 3; i++) {
      t[i] = T(t_c[i]);
    }
    scale = T(scale_c);
    log_Sim3(R, t, scale, params);
  }
  explicit Sim3_t(const Eigen::Matrix4d& pose) {
    const Eigen::Matrix3d scale_rot = pose.block<3, 3>(0, 0);
    const double _scale     = std::pow(scale_rot.determinant(), 1.0 / 3);
    const Eigen::Matrix3d rot       = 1 / _scale * scale_rot;

    T R[9];
    T t[3];
    T scale;

    for (uint32_t iy = 0; iy < 3; iy++) {
      for (uint32_t ix = 0; ix < 3; ix++) {
        R[3 * ix + iy] = T(rot(iy, ix));
      }
      t[iy] = T(pose(iy, 3));
    }
    scale = T(_scale);
    log_Sim3(R, t, scale, params);
  }

  Sim3_t inverse() const {
    T R[9];
    T t[3];
    T scale;
    exp_Sim3(params, R, t, &scale);
    T new_R[9];
    for (uint32_t iy = 0; iy < 3; iy++) {
      for (uint32_t ix = 0; ix < 3; ix++) {
        new_R[ix * 3 + iy] = R[iy * 3 + ix];
      }
    }
    T new_scale = T(1.0) / scale;
    T new_t[3];
    multi_vec(new_R, t, new_t);
    for (auto& i : new_t) {
      i /= -scale;
    }
    return Sim3_t(new_R, new_t, new_scale);
  }

  Sim3_t operator*(const Sim3_t& rhs) const {
    T R_l[9];
    T t_l[3];
    T s_l;
    exp_Sim3(params, R_l, t_l, &s_l);

    T R_r[9];
    T t_r[3];
    T s_r;
    exp_Sim3(rhs.params, R_r, t_r, &s_r);

    T new_R[9];
    multi_mat(R_l, R_r, new_R);
    T new_scale = s_l * s_r;
    T new_t[3];
    multi_vec(R_l, t_r, new_t);
    for (uint32_t i = 0; i < 3; i++) {
      new_t[i] = s_l * new_t[i] + t_l[i];
    }
    return Sim3_t(new_R, new_t, new_scale);
  }

  Eigen::Matrix4d as_eigen_mat() const {
    T R[9];
    T t[3];
    T scale;
    exp_Sim3(params, R, t, &scale);

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    double s = convert_to_double(scale);
    for (uint32_t iy = 0; iy < 3; iy++) {
      for (uint32_t ix = 0; ix < 3; ix++) {
        pose(iy, ix) = s * convert_to_double(R[3 * ix + iy]);
      }
      pose(iy, 3) = convert_to_double(t[iy]);
    }
    return pose;
  }
};
} // namespace pgo_core
} // namespace bundlefit
#endif // BUNDLEFIT_SIM3_H
