#ifndef LIE_SIM3_CACHE_H
#define LIE_SIM3_CACHE_H

#include <cstddef> // size_t
#include <cstdint> // uint64_t

namespace lie::Sim3 {
class Sim3Cache {
  union FloatToBinary {
    double fp;
    uint64_t bin;
  };
  struct RecordKey {
    RecordKey()
        : has_value(false) {}
    RecordKey(const double* _params)
        : has_value(true) {
      for (size_t i = 0; i < 6; i++) {
        params[i] = _params[i];
      }
    }
    bool has_value;
    double params[6];

    bool test(const double* _params) const {
      if (!has_value) {
        return false;
      }
      for (size_t i = 0; i < 6; i++) {
        if (FloatToBinary{params[i]}.bin != FloatToBinary{_params[i]}.bin) {
          return false;
        }
      }
      return true;
    }
  };

  mutable bool enable_ = true;

  // mutable RecordKey prev_key;
  mutable double R_[9];
  mutable double inv_theta_sq_ = 0;
  mutable double theta_        = 0;
  mutable double scale_        = 0;
  mutable double B_[9];

  mutable bool cache_R_avail_ = false;
  mutable bool cache_B_avail_ = false;
  mutable RecordKey prev_key_R_;
  mutable RecordKey prev_key_B_;

public:
  Sim3Cache() = default;
  ~Sim3Cache() {}

  void set_enable(const bool enable) { enable_ = enable; }
  bool is_enable() const { return enable_; }

  void clear() const {
    cache_R_avail_ = false;
    cache_B_avail_ = false;
  }

  bool has_R_cache() const { return cache_R_avail_; }
  bool has_B_cache() const { return cache_B_avail_; }

  void get_R([[maybe_unused]] const double* key_params, double** R,
             double* _inv_theta_sq, double* _theta, double* _scale) const {
#ifdef CHECK_Sim3_CACHE
    if (!prev_key_R_.test(key_params)) {
      throw std::runtime_error(
          "Invalid cache access, cache should be cleared before optimize "
          "step.");
    }
#endif // CHECK_Sim3_CACHE
    *R             = R_;
    *_inv_theta_sq = inv_theta_sq_;
    *_theta        = theta_;
    *_scale        = scale_;
  }
  void get_B([[maybe_unused]] const double* key_params, double** B) const {
#ifdef CHECK_Sim3_CACHE
    if (!prev_key_B_.test(key_params)) {
      throw std::runtime_error(
          "Invalid cache access, cache should be cleared before optimize "
          "step.");
    }
#endif // CHECK_Sim3_CACHE
    *B = B_;
  }
  void store_R([[maybe_unused]] const double* key_params, const double R[9],
               const double _inv_theta_sq, const double _theta,
               const double _scale) const {
#ifdef CHECK_Sim3_CACHE
    if (cache_R_avail_) {
      if (!prev_key_R_.test(key_params)) {
        throw std::runtime_error(
            "Invalid overwriting to lie::Sim3::Sim3Cache.");
      }
    }
    prev_key_R_ = RecordKey(key_params);
#endif // CHECK_Sim3_CACHE
    cache_R_avail_ = true;

    for (int i = 0; i < 9; i++) {
      R_[i] = R[i];
    }
    inv_theta_sq_ = _inv_theta_sq;
    theta_        = _theta;
    scale_        = _scale;
  }
  void store_B([[maybe_unused]] const double* key_params, double B[9]) const {
#ifdef CHECK_Sim3_CACHE
    if (cache_B_avail_) {
      if (!prev_key_B_.test(key_params)) {
        throw std::runtime_error(
            "Invalid overwriting to lie::Sim3::Sim3Cache.");
      }
    }
    prev_key_B_ = RecordKey(key_params);
#endif // CHECK_Sim3_CACHE
    cache_B_avail_ = true;

    for (int i = 0; i < 9; i++) {
      B_[i] = B[i];
    }
  }
};
} // namespace lie::Sim3
#endif // LIE_SIM3_CACHE_H
