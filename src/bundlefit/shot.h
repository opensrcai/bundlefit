#ifndef BUNDLEFIT_SHOT_H
#define BUNDLEFIT_SHOT_H

#include <memory>
#include <unordered_set>

#include "lie/sim3_cache.h"
#include "bundlefit/types.h"

namespace bundlefit {

struct ReprojectionError;
// Shot struct
struct Shot {
  enum class Type {
    SE3 = 0,
    Sim3,
    None
  };
  static constexpr auto NUM_PARAMS = make_const_dict<Type, uint32_t>(
      {{Type::SE3, 6}, {Type::Sim3, 7}, {Type::None, 0}});
  static constexpr auto MAX_NUM_PARAMS = NUM_PARAMS.max();

  Shot(const uint32_t _id, const Eigen::Matrix<double, 6, 1>& _params, const bool _is_fixed,
       const bool _use_Sim3_cache)
      : id(_id)
      , type(Type::SE3)
      , params(new double[NUM_PARAMS[type]])
      , is_fixed(_is_fixed) {
    for (uint32_t i = 0; i < NUM_PARAMS[type]; i++) {
      params[i] = _params(i);
    }
    if (_use_Sim3_cache) {
      cache = std::make_unique<lie::Sim3::Sim3Cache>();
    }
  }
  Shot(const uint32_t _id, const Eigen::Matrix<double, 7, 1>& _params, const bool _is_fixed,
       const bool _use_Sim3_cache)
      : id(_id)
      , type(Type::Sim3)
      , params(new double[NUM_PARAMS[type]])
      , is_fixed(_is_fixed) {
    for (uint32_t i = 0; i < NUM_PARAMS[type]; i++) {
      params[i] = _params(i);
    }
    if (_use_Sim3_cache) {
      cache = std::make_unique<lie::Sim3::Sim3Cache>();
    }
  }
  ~Shot() { delete[] params; }
  const uint32_t id;
  const Type type;
  double* params;
  bool is_fixed;
  std::unordered_set<ReprojectionError*> residuals;

  std::unique_ptr<lie::Sim3::Sim3Cache> cache;
};
} // namespace bundlefit
#endif // BUNDLEFIT_SHOT_H
