#ifndef LIE_LIE_3D_JAC_H
#define LIE_LIE_3D_JAC_H

#include <unordered_map>

#include "lie/sim3_cache.h"

namespace lie {

namespace Sim3 {
class Sim3Cache;
}

namespace {
template <size_t num_params, size_t N>
inline void sim3_map(const double* const v, const double* const pos_w,
                     double* pos_l, const bool derive_shot, double dpl_ds[3][N],
                     const bool derive_point, double dpl_dpw[3][3],
                     const Sim3::Sim3Cache* cache = nullptr);
template <size_t num_params, size_t N>
inline void sim3_inv_map(const double* const v, const double* const pos_w,
                         double* pos_l, const bool derive_shot,
                         double dpl_ds[3][N], const bool derive_point,
                         double dpl_dpw[3][3]);
} // namespace

namespace SE3 {

template <size_t N>
inline void map(const double* const v, const double* const pos_w, double* pos_l,
                const bool derive_shot, double dpl_ds[3][N],
                const bool derive_point, double dpl_dpw[3][3],
                const Sim3::Sim3Cache* cache = nullptr) {
  sim3_map<6, N>(v, pos_w, pos_l, derive_shot, dpl_ds, derive_point, dpl_dpw,
                 cache);
}

template <size_t N>
inline void inv_map(const double* const v, const double* const pos_l,
                    double* pos_w, const bool derive_shot, double dpl_ds[3][N],
                    const bool derive_point, double dpl_dpw[3][3]) {
  sim3_inv_map<6, N>(v, pos_l, pos_w, derive_shot, dpl_ds, derive_point,
                     dpl_dpw);
}

} // namespace SE3

namespace Sim3 {
template <size_t N>
inline void map(const double* const v, const double* const pos_w, double* pos_l,
                const bool derive_shot, double dpl_ds[3][N],
                const bool derive_point, double dpl_dpw[3][3],
                const Sim3::Sim3Cache* cache = nullptr) {
  sim3_map<7, N>(v, pos_w, pos_l, derive_shot, dpl_ds, derive_point, dpl_dpw,
                 cache);
}
template <size_t N>
inline void inv_map(const double* const v, const double* const pos_l,
                    double* pos_w, const bool derive_shot, double dpl_ds[3][N],
                    const bool derive_point, double dpl_dpw[3][3]) {
  sim3_inv_map<7, N>(v, pos_l, pos_w, derive_shot, dpl_ds, derive_point,
                     dpl_dpw);
}

} // namespace Sim3

namespace {
// Transform 3D point to 3D point by shot pose parameter (analytical
// differentiation) According to the Jacobian derivation for SO(3) from eq.(8)
// in https://arxiv.org/pdf/1312.0788.pdf
template <size_t num_params, size_t N>
inline void sim3_map(const double* const v, const double* const pos_w,
                     double* pos_l, const bool derive_shot, double dpl_ds[3][N],
                     const bool derive_point, double dpl_dpw[3][3],
                     const Sim3::Sim3Cache* cache) {
  static_assert((num_params == 6 || num_params == 7),
                "num_params should be 6 or 7");
  static_assert(N >= num_params, "Too small Jacobian size");

  const bool cache_available = cache && cache->is_enable();

  // SE(3) transform
  double R_data[9];
  double* R_head      = R_data;
  auto R              = reinterpret_cast<double(*)[3]>(R_head);
  double inv_theta_sq = 0;
  double theta        = 0;

  // Scale for Sim(3) transform, transform is equivalent SE(3) when scale = 1.0
  double scale = 1.0;

  {
    if (cache_available && cache->has_R_cache()) {
      cache->get_R(v, &R_head, &inv_theta_sq, &theta, &scale);
      R = reinterpret_cast<double(*)[3]>(R_head);
    }
    else {
      const double theta_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      inv_theta_sq          = 1.0 / theta_sq;
      theta                 = std::sqrt(theta_sq);

      const double cos = std::cos(theta);
      const double sin = std::sin(theta);
      const double alpha
          = theta < 1e-8 ? 0.5 : (1 - cos) * inv_theta_sq;  // (1-cosθ) / θ^2
      const double beta = theta < 1e-8 ? 1.0 : sin / theta; // sinθ / θ
      R[0][0]           = cos + v[0] * v[0] * alpha;
      R[0][1]           = v[0] * v[1] * alpha - v[2] * beta;
      R[0][2]           = v[0] * v[2] * alpha + v[1] * beta;
      R[1][0]           = v[1] * v[0] * alpha + v[2] * beta;
      R[1][1]           = cos + v[1] * v[1] * alpha;
      R[1][2]           = v[1] * v[2] * alpha - v[0] * beta;
      R[2][0]           = v[2] * v[0] * alpha - v[1] * beta;
      R[2][1]           = v[2] * v[1] * alpha + v[0] * beta;
      R[2][2]           = cos + v[2] * v[2] * alpha;

      if (num_params == 7) {
        scale = std::exp(v[6]);
      }

      if (cache_available) {
        cache->store_R(v, R_head, inv_theta_sq, theta, scale);
      }
    }
  }

  const double x = pos_w[0];
  const double y = pos_w[1];
  const double z = pos_w[2];

  // Transformation from world to local
  pos_l[0] = scale * (R[0][0] * x + R[0][1] * y + R[0][2] * z) + v[3];
  pos_l[1] = scale * (R[1][0] * x + R[1][1] * y + R[1][2] * z) + v[4];
  pos_l[2] = scale * (R[2][0] * x + R[2][1] * y + R[2][2] * z) + v[5];

  if (derive_point) {
    for (uint32_t i = 0; i < 3; i++) {
      for (uint32_t j = 0; j < 3; j++) {
        dpl_dpw[i][j] = scale * R[i][j];
      }
    }
  }

  if (derive_shot) {
    // Jacobian for translation is identity
    dpl_ds[0][3] = 1.0;
    dpl_ds[0][4] = 0.0;
    dpl_ds[0][5] = 0.0;
    dpl_ds[1][3] = 0.0;
    dpl_ds[1][4] = 1.0;
    dpl_ds[1][5] = 0.0;
    dpl_ds[2][3] = 0.0;
    dpl_ds[2][4] = 0.0;
    dpl_ds[2][5] = 1.0;

    // Jacobian for scale parameter
    if (num_params == 7) {
      dpl_ds[0][6] = scale * (R[0][0] * x + R[0][1] * y + R[0][2] * z);
      dpl_ds[1][6] = scale * (R[1][0] * x + R[1][1] * y + R[1][2] * z);
      dpl_ds[2][6] = scale * (R[2][0] * x + R[2][1] * y + R[2][2] * z);
    }

    // Jacobian for rotation around origin
    if (theta < 1e-8) {
      dpl_ds[0][0] = 0.0;
      dpl_ds[0][1] = scale * z;
      dpl_ds[0][2] = -scale * y;
      dpl_ds[1][0] = -scale * z;
      dpl_ds[1][1] = 0.0;
      dpl_ds[1][2] = scale * x;
      dpl_ds[2][0] = scale * y;
      dpl_ds[2][1] = -scale * x;
      dpl_ds[2][2] = 0.0;
    }
    else {
      // correspond to -R[u]x (former of eq.(8)) in paper
      const double A1 = R[0][2] * y - R[0][1] * z;
      const double A2 = R[0][0] * z - R[0][2] * x;
      const double A3 = R[0][1] * x - R[0][0] * y;
      const double A4 = R[1][2] * y - R[1][1] * z;
      const double A5 = R[1][0] * z - R[1][2] * x;
      const double A6 = R[1][1] * x - R[1][0] * y;
      const double A7 = R[2][2] * y - R[2][1] * z;
      const double A8 = R[2][0] * z - R[2][2] * x;
      const double A9 = R[2][1] * x - R[2][0] * y;

      // correspond to {vv^T + (R^T - Id)[v]x } / |v|^2 (latter fraction of
      // eq.(8)) in paper
      /*
      const double B1 = (v[0] * v[0] + R[1][0] * v[2] - R[2][0] * v[1]) *
      inv_theta_sq; const double B2 = (v[0] * v[1] + R[2][0] * v[0] - (R[0][0]
      - 1.0) * v[2]) * inv_theta_sq; const double B3 = (v[0] * v[2] + (R[0][0]
      - 1.0) * v[1] - R[1][0] * v[0]) * inv_theta_sq; const double B4 = (v[1] *
      v[0] + (R[1][1] - 1.0) * v[2] - R[2][1] * v[1]) * inv_theta_sq; const
      double B5 = (v[1] * v[1] + R[2][1] * v[0] - R[0][1] * v[2]) *
      inv_theta_sq; const double B6 = (v[1] * v[2] + R[0][1] * v[1] - (R[1][1]
      - 1.0) * v[0]) * inv_theta_sq; const double B7 = (v[2] * v[0] + R[1][2] *
      v[2] - (R[2][2] - 1.0) * v[1]) * inv_theta_sq; const double B8 = (v[2] *
      v[1] + (R[2][2] - 1.0) * v[0] - R[0][2] * v[2]) * inv_theta_sq; const
      double B9 = (v[2] * v[2] + R[0][2] * v[1] - R[1][2] * v[0]) *
      inv_theta_sq;
       */

      double B_data[9];
      double* B_head = B_data;
      auto B         = reinterpret_cast<double(*)[3]>(B_head);
      if (cache_available && cache->has_B_cache()) {
        cache->get_B(v, &B_head);
        B = reinterpret_cast<double(*)[3]>(B_head);
      }
      else {
        B[0][0]
            = (v[0] * v[0] + R[1][0] * v[2] - R[2][0] * v[1]) * inv_theta_sq;
        B[0][1] = (v[0] * v[1] + R[2][0] * v[0] - (R[0][0] - 1.0) * v[2])
                  * inv_theta_sq;
        B[0][2] = (v[0] * v[2] + (R[0][0] - 1.0) * v[1] - R[1][0] * v[0])
                  * inv_theta_sq;
        B[1][0] = (v[1] * v[0] + (R[1][1] - 1.0) * v[2] - R[2][1] * v[1])
                  * inv_theta_sq;
        B[1][1]
            = (v[1] * v[1] + R[2][1] * v[0] - R[0][1] * v[2]) * inv_theta_sq;
        B[1][2] = (v[1] * v[2] + R[0][1] * v[1] - (R[1][1] - 1.0) * v[0])
                  * inv_theta_sq;
        B[2][0] = (v[2] * v[0] + R[1][2] * v[2] - (R[2][2] - 1.0) * v[1])
                  * inv_theta_sq;
        B[2][1] = (v[2] * v[1] + (R[2][2] - 1.0) * v[0] - R[0][2] * v[2])
                  * inv_theta_sq;
        B[2][2]
            = (v[2] * v[2] + R[0][2] * v[1] - R[1][2] * v[0]) * inv_theta_sq;
        if (cache_available) {
          cache->store_B(v, B_data);
        }
      }

      dpl_ds[0][0] = scale * (A1 * B[0][0] + A2 * B[1][0] + A3 * B[2][0]);
      dpl_ds[0][1] = scale * (A1 * B[0][1] + A2 * B[1][1] + A3 * B[2][1]);
      dpl_ds[0][2] = scale * (A1 * B[0][2] + A2 * B[1][2] + A3 * B[2][2]);
      dpl_ds[1][0] = scale * (A4 * B[0][0] + A5 * B[1][0] + A6 * B[2][0]);
      dpl_ds[1][1] = scale * (A4 * B[0][1] + A5 * B[1][1] + A6 * B[2][1]);
      dpl_ds[1][2] = scale * (A4 * B[0][2] + A5 * B[1][2] + A6 * B[2][2]);
      dpl_ds[2][0] = scale * (A7 * B[0][0] + A8 * B[1][0] + A9 * B[2][0]);
      dpl_ds[2][1] = scale * (A7 * B[0][1] + A8 * B[1][1] + A9 * B[2][1]);
      dpl_ds[2][2] = scale * (A7 * B[0][2] + A8 * B[1][2] + A9 * B[2][2]);
    }
  }
}

// Transform 3D point to 3D point by inverse of shot pose parameter (analytical
// differentiation) When v-derivative for y = R(v)x is known as ∂y/∂v,
// v-derivative of x = R(-v)y should be ∂x/∂v = -∂y/∂w|w=-v
template <size_t num_params, size_t N>
inline void sim3_inv_map(const double* const v, const double* const pos_l,
                         double* pos_w, const bool derive_shot,
                         double dpl_ds[3][N], const bool derive_point,
                         double dpl_dpw[3][3]) {
  static_assert((num_params == 6 || num_params == 7),
                "num_params should be 6 or 7");
  static_assert(N >= num_params, "Too small Jacobian size");

  const double theta_sq     = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const double inv_theta_sq = 1.0 / theta_sq;
  const double theta        = std::sqrt(theta_sq);
  const double cos          = std::cos(theta);
  const double sin          = std::sin(theta);
  const double alpha
      = theta < 1e-8 ? 0.5 : (1 - cos) * inv_theta_sq; // (1-cosθ) / θ^2
  const double beta = theta < 1e-8 ? 1 : sin / theta;  // sinθ / θ
  const double x    = pos_l[0] - v[3];
  const double y    = pos_l[1] - v[4];
  const double z    = pos_l[2] - v[5];

  // Scale for Sim(3) transform, transform is equivalent SE(3) when scale = 1.0
  double inv_scale = 1.0;
  if (num_params == 7) {
    inv_scale = 1 / std::exp(v[6]);
  }

  // Rotation matrix
  double R[3][3];
  R[0][0] = cos + v[0] * v[0] * alpha;
  R[0][1] = v[0] * v[1] * alpha - v[2] * beta;
  R[0][2] = v[0] * v[2] * alpha + v[1] * beta;
  R[1][0] = v[1] * v[0] * alpha + v[2] * beta;
  R[1][1] = cos + v[1] * v[1] * alpha;
  R[1][2] = v[1] * v[2] * alpha - v[0] * beta;
  R[2][0] = v[2] * v[0] * alpha - v[1] * beta;
  R[2][1] = v[2] * v[1] * alpha + v[0] * beta;
  R[2][2] = cos + v[2] * v[2] * alpha;

  // Transformation from local to world (Apply inverse of R)
  pos_w[0] = inv_scale * (R[0][0] * x + R[1][0] * y + R[2][0] * z);
  pos_w[1] = inv_scale * (R[0][1] * x + R[1][1] * y + R[2][1] * z);
  pos_w[2] = inv_scale * (R[0][2] * x + R[1][2] * y + R[2][2] * z);

  if (derive_point) {
    for (uint32_t i = 0; i < 3; i++) {
      for (uint32_t j = 0; j < 3; j++) {
        dpl_dpw[i][j] = inv_scale * R[j][i];
      }
    }
  }

  if (derive_shot) {
    // Jacobian for translation is identity
    for (uint32_t i = 0; i < 3; i++) {
      for (uint32_t j = 0; j < 3; j++) {
        dpl_ds[i][j + 3] = -inv_scale * R[j][i];
      }
    }

    // Jacobian for scale parameter
    if (num_params == 7) {
      dpl_ds[0][6] = -pos_w[0];
      dpl_ds[1][6] = -pos_w[1];
      dpl_ds[2][6] = -pos_w[2];
    }

    // Jacobian for rotation around origin
    if (theta < 1e-8) {
      dpl_ds[0][0] = 0.0;
      dpl_ds[0][1] = -inv_scale * z;
      dpl_ds[0][2] = inv_scale * y;
      dpl_ds[1][0] = inv_scale * z;
      dpl_ds[1][1] = 0.0;
      dpl_ds[1][2] = -inv_scale * x;
      dpl_ds[2][0] = -inv_scale * y;
      dpl_ds[2][1] = inv_scale * x;
      dpl_ds[2][2] = 0.0;
    }
    else {
      // R[u]^T x
      const double A1 = R[1][0] * z - R[2][0] * y;
      const double A2 = R[2][0] * x - R[0][0] * z;
      const double A3 = R[0][0] * y - R[1][0] * x;
      const double A4 = R[1][1] * z - R[2][1] * y;
      const double A5 = R[2][1] * x - R[0][1] * z;
      const double A6 = R[0][1] * y - R[1][1] * x;
      const double A7 = R[1][2] * z - R[2][2] * y;
      const double A8 = R[2][2] * x - R[0][2] * z;
      const double A9 = R[0][2] * y - R[1][2] * x;

      // {vv^T - (R - Id)[v]x } / |v|^2
      const double B1
          = (v[0] * v[0] - R[0][1] * v[2] + R[0][2] * v[1]) * inv_theta_sq;
      const double B2 = (v[0] * v[1] - R[0][2] * v[0] + (R[0][0] - 1.0) * v[2])
                        * inv_theta_sq;
      const double B3 = (v[0] * v[2] - (R[0][0] - 1.0) * v[1] + R[0][1] * v[0])
                        * inv_theta_sq;
      const double B4 = (v[1] * v[0] - (R[1][1] - 1.0) * v[2] + R[1][2] * v[1])
                        * inv_theta_sq;
      const double B5
          = (v[1] * v[1] - R[1][2] * v[0] + R[1][0] * v[2]) * inv_theta_sq;
      const double B6 = (v[1] * v[2] - R[1][0] * v[1] + (R[1][1] - 1.0) * v[0])
                        * inv_theta_sq;
      const double B7 = (v[2] * v[0] - R[2][1] * v[2] + (R[2][2] - 1.0) * v[1])
                        * inv_theta_sq;
      const double B8 = (v[2] * v[1] - (R[2][2] - 1.0) * v[0] + R[2][0] * v[2])
                        * inv_theta_sq;
      const double B9
          = (v[2] * v[2] - R[2][0] * v[1] + R[2][1] * v[0]) * inv_theta_sq;

      dpl_ds[0][0] = inv_scale * (A1 * B1 + A2 * B4 + A3 * B7);
      dpl_ds[0][1] = inv_scale * (A1 * B2 + A2 * B5 + A3 * B8);
      dpl_ds[0][2] = inv_scale * (A1 * B3 + A2 * B6 + A3 * B9);
      dpl_ds[1][0] = inv_scale * (A4 * B1 + A5 * B4 + A6 * B7);
      dpl_ds[1][1] = inv_scale * (A4 * B2 + A5 * B5 + A6 * B8);
      dpl_ds[1][2] = inv_scale * (A4 * B3 + A5 * B6 + A6 * B9);
      dpl_ds[2][0] = inv_scale * (A7 * B1 + A8 * B4 + A9 * B7);
      dpl_ds[2][1] = inv_scale * (A7 * B2 + A8 * B5 + A9 * B8);
      dpl_ds[2][2] = inv_scale * (A7 * B3 + A8 * B6 + A9 * B9);
    }
  }
}
} // namespace
} // namespace lie
#endif // LIE_LIE_3D_JAC_H
