#ifndef LIE_LIE_3D_CERES_H
#define LIE_LIE_3D_CERES_H

#include <ceres/jet.h>
#include <ceres/rotation.h>

#include "lie/lie_3d.h"

namespace lie {

namespace SE3 {

// Transform an 3D point in world coords to local coords
template <typename T>
inline void map(const T* const params, const T* const pos_w, T* pos_l) {
  ceres::AngleAxisRotatePoint(params, pos_w, pos_l);
  for (uint32_t i = 0; i < 3; i++) {
    pos_l[i] += params[3 + i];
  }
}

// Inverse function of map
// Transform an 3D point in local coords to world coords
template <typename T>
inline void inv_map(const T* const params, const T* const pos_l, T* pos_w) {
  T aux[3];
  for (uint32_t i = 0; i < 3; i++) {
    aux[i] = pos_l[i] - params[3 + i];
  }

  T inv_rot[3];
  for (uint32_t i = 0; i < 3; i++) {
    inv_rot[i] = -params[i];
  }
  ceres::AngleAxisRotatePoint(inv_rot, aux, pos_w);
}

} // namespace SE3

namespace Sim3 {
// Function for Sim(3) transform
template <typename T>
inline void map(const T* const params, const T* const point_pos, T* pos_l) {
  ceres::AngleAxisRotatePoint(params, point_pos, pos_l);

  T scale = ceres::exp(params[6]);
  for (uint32_t i = 0; i < 3; i++) {
    pos_l[i] *= scale;
  }

  for (uint32_t i = 0; i < 3; i++) {
    pos_l[i] += params[3 + i];
  }
}

// Function for Sim(3) inverse transform
template <typename T>
inline void inv_map(const T* const params, const T* const pos_l, T* pos_w) {
  T aux[3];
  for (uint32_t i = 0; i < 3; i++) {
    aux[i] = pos_l[i] - params[3 + i];
  }

  T inv_rot[3];
  for (uint32_t i = 0; i < 3; i++) {
    inv_rot[i] = -params[i];
  }
  ceres::AngleAxisRotatePoint(inv_rot, aux, pos_w);
  T scale = ceres::exp(params[6]);
  for (uint32_t i = 0; i < 3; i++) {
    pos_w[i] /= scale;
  }
}

} // namespace Sim3
} // namespace lie

#endif // LIE_LIE_3D_CERES_H
