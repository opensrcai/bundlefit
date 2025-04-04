#ifndef BUNDLEFIT_LANDMARK_H
#define BUNDLEFIT_LANDMARK_H

#include <unordered_set>

namespace bundlefit {
struct Shot;
struct ReprojectionError;

// Landmark struct
struct Landmark {
  enum class Type
  {
    Point3D = 0
  };
  static constexpr auto NUM_PARAMS
      = make_const_dict<Type, uint32_t>({{Type::Point3D, 3}});
  static constexpr auto MAX_NUM_PARAMS = NUM_PARAMS.max();

  Landmark(const uint32_t _id, const Eigen::Vector3d& position, const bool _is_fixed,
           const Shot* _local_coord_system = nullptr)
      : id(_id)
      , type(Type::Point3D)
      , params(new double[NUM_PARAMS[type]])
      , is_fixed(_is_fixed)
      , local_coord_system(_local_coord_system) {
    for (uint32_t i = 0; i < NUM_PARAMS[type]; i++) {
      params[i] = position[i];
    }
  }
  ~Landmark() { delete[] params; }
  const uint32_t id;
  const Type type = Type::Point3D;
  double* params;
  const bool is_fixed;

  const Shot* local_coord_system;
  std::unordered_set<ReprojectionError*> residuals;
};

} // namespace bundlefit

#endif // BUNDLEFIT_LANDMARK_H
