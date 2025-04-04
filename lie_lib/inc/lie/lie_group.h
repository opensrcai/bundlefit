#ifndef LIE_TYPES_H
#define LIE_TYPES_H

#include <lie/lie_3d.h>

namespace lie {

namespace AffineType {

class SO3 {
public:
  static constexpr uint32_t id         = 0;
  static constexpr uint32_t num_params = 3;
};
class SE3 {
public:
  static constexpr uint32_t id         = 1;
  static constexpr uint32_t num_params = 6;
};
class Sim3 {
public:
  static constexpr uint32_t id         = 2;
  static constexpr uint32_t num_params = 7;
};

} // namespace AffineType

double scale_of_sR(const Eigen::Matrix3d& sR);

template <typename T>
class LieGroup_t;
using SO3_t  = LieGroup_t<AffineType::SO3>;
using SE3_t  = LieGroup_t<AffineType::SE3>;
using Sim3_t = LieGroup_t<AffineType::Sim3>;

template <typename T2>
static Sim3_t operator*(const Sim3_t& lhs, const LieGroup_t<T2>& rhs);
template <typename T1>
static Sim3_t operator*(const LieGroup_t<T1>& lhs, const Sim3_t& rhs);

// LieGroup_t represents several Lie Groups for 3D space such as SE(3).
// It can be constructed from Matrix, lie algebra parameters.
template <typename T>
class LieGroup_t {
  friend class LieGroup_t<AffineType::SO3>;
  friend class LieGroup_t<AffineType::SE3>;
  friend class LieGroup_t<AffineType::Sim3>;

  template <typename T2>
  friend Sim3_t operator*(const Sim3_t& lhs, const LieGroup_t<T2>& rhs);

  template <typename T1>
  friend Sim3_t operator*(const LieGroup_t<T1>& lhs, const Sim3_t& rhs);

public:
  // Check given rotation matrix satisfies the property of SO(3)
  static inline bool is_SO3(const Eigen::Matrix3d& R) {
    // Check the determinant
    const bool det_ok = std::abs(scale_of_sR(R) - 1.0) < 1e-6;
    // Check if the rotation matrix is orthogonal matrix
    const bool ortho_ok
        = (R * R.transpose() - Eigen::Matrix3d::Identity()).squaredNorm()
          < 1e-12;
    return det_ok && ortho_ok;
  }

  // Check given rotation matrix is scaled SO(3)
  static inline bool is_SO3_with_scale(const Eigen::Matrix3d& R) {
    const auto s = scale_of_sR(R);
    if (s <= 0) {
      return false;
    }
    return is_SO3(R / s);
  }

private:
  // Construct from lie-algebra parameters
  void _set_SO3(const Eigen::Vector3d& params) {
    R_ = lie::SO3::exp(params);
    t_ = Eigen::Vector3d::Zero();
  }
  void _set_SE3(const Eigen::Matrix<double, 6, 1>& params) {
    const auto Rt = lie::SE3::exp(params);
    R_            = Rt.first;
    t_            = Rt.second;
  }
  void _set_Sim3(const Eigen::Matrix<double, 7, 1>& params) {
    const auto Rts = lie::Sim3::exp(params);
    R_             = std::get<0>(Rts) * std::get<2>(Rts);
    t_             = std::get<1>(Rts);
  }

  // Internal constructors for unpacking parameter list
  void _set_SO3(const Eigen::Matrix3d& R) {
    assert(is_SO3(R) && "Given R is not SO(3) matrix.");
    R_ = R;
    t_ = Eigen::Vector3d::Zero();
  }
  void _set_SE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    assert(is_SO3(R) && "Given R is not SO(3) matrix.");
    R_ = R;
    t_ = t;
  }
  void _set_Sim3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    assert(is_SO3_with_scale(R) && "Given R is not scaled SO(3) matrix.");
    R_ = R;
    t_ = t;
  }
  void _set_Sim3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                 const double s) {
    assert(is_SO3(R) && "Given R is not SO(3) matrix.");
    R_ = R * s;
    t_ = t;
  }

  inline double _inv_sq_scale() const {
    return 3.0 / ((R_ * R_.transpose()).trace());
  }

public:
  // Construct as Identical for default
  LieGroup_t()
      : R_(Eigen::Matrix3d::Identity())
      , t_(Eigen::Vector3d::Zero()) {}

  // Trivial copy constructor
  LieGroup_t(const LieGroup_t& obj)
      : R_(obj.R_)
      , t_(obj.t_) {}

  // Construct from another LieGroup_t
  // Removing parameter is not allowed (such as Sim3 -> SE3 conversion)
  template <typename T2>
  explicit LieGroup_t(const LieGroup_t<T2>& obj)
      : R_(obj.R_)
      , t_(obj.t_) {
    static_assert(T::id >= T2::id, "Cast with dropping parameter is invalid.");
  }

  // Construct from 4x4 homogeneous matrix
  // The matrix should be this form : | R  t |
  //                                  | 0  1 |
  explicit LieGroup_t(const Eigen::Matrix4d& mat_hg) {
    assert(mat_hg.block(3, 0, 1, 3).squaredNorm() < 1e-12
           && "Given matrix is not homogeneous.");
    assert(std::abs(mat_hg(3, 3) - 1.0) < 1e-12
           && "Given matrix is not homogeneous.");
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      assert(mat_hg.block(0, 3, 3, 1).squaredNorm() < 1e-12
             && "Given matrix is not SO(3), it has translation parameters.");
      R_ = mat_hg.block<3, 3>(0, 0);
      t_ = Eigen::Vector3d::Zero();
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value) {
      R_ = mat_hg.block<3, 3>(0, 0);
      assert(is_SO3(R_) && "R part of given matrix is not SO(3).");
      t_ = mat_hg.block<3, 1>(0, 3);
    }
    if constexpr (std::is_same<T, AffineType::Sim3>::value) {
      R_ = mat_hg.block<3, 3>(0, 0);
      assert(is_SO3_with_scale(R_)
             && "R part of given matrix is not scaled SO(3).");
      t_ = mat_hg.block<3, 1>(0, 3);
    }
  }

  // Constructor for different number of arguments
  // like: SO3_t(R), SE_t(R, t), Sim3_t(R, t, s).
  template <class... Args>
  LieGroup_t(Args... args) {
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      _set_SO3(args...);
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value) {
      _set_SE3(args...);
    }
    if constexpr (std::is_same<T, AffineType::Sim3>::value) {
      _set_Sim3(args...);
    }
  }

  // Get inverse of transform
  inline LieGroup_t inverse() const {
    const Eigen::Matrix3d R_T = R_.transpose();
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      return {R_T};
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value) {
      return {R_T, -R_T * t_};
    }
    if constexpr (std::is_same<T, AffineType::Sim3>::value) {
      const Eigen::Matrix3d rot_inv = R_T * _inv_sq_scale();
      return {rot_inv, -rot_inv * t_};
    }
  }

  // Map a 3D vector by this transformation
  inline Eigen::Vector3d operator*(const Eigen::Vector3d& pos_l) const {
    return R_ * pos_l + t_;
  }

  // Multiply two LieGroups
  LieGroup_t operator*(const LieGroup_t& rhs) const {
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      return Eigen::Matrix3d{R_ * rhs.R_};
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value
                  || std::is_same<T, AffineType::Sim3>::value) {
      return {R_ * rhs.R_, R_ * rhs.t_ + t_};
    }
  }

  // Get rotation component
  // *** If this is Sim3_t, this is different to top left 3x3 matrix ***
  inline const Eigen::Matrix3d R() const {
    if constexpr (std::is_same<T, AffineType::SO3>::value
                  || std::is_same<T, AffineType::SE3>::value) {
      return R_;
    }
    else {
      return R_ / scale_of_sR(R_);
    }
  }

  // Get translation component
  inline const Eigen::Vector3d t() const { return t_; }

  // Get scale component
  inline double s() const { return scale_of_sR(R_); }

  // Get log (lie algebra of lie group) parameters
  Vec_t<T::num_params> log() const {
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      return lie::SO3::log(R());
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value) {
      return lie::SE3::log(R(), t());
    }
    if constexpr (std::is_same<T, AffineType::Sim3>::value) {
      return lie::Sim3::log(R(), t(), s());
    }
  }

  // Get camera center of local coordinate
  inline Eigen::Vector3d cam_center() const {
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      return {0, 0, 0};
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value) {
      return -R_.transpose() * t_;
    }
    if constexpr (std::is_same<T, AffineType::Sim3>::value) {
      return -R_.transpose() * _inv_sq_scale() * t_;
    }
  }

  // Get transformation as a homogeneous matrix of this LieGroup
  Eigen::Matrix4d homogeneous() const {
    Eigen::Matrix4d hg   = Eigen::Matrix4d::Identity();
    hg.block<3, 3>(0, 0) = R_;
    hg.block<3, 1>(0, 3) = t_;
    return hg;
  }

  static LieGroup_t Identity() {
    if constexpr (std::is_same<T, AffineType::SO3>::value) {
      return Eigen::Vector3d{0, 0, 0};
    }
    if constexpr (std::is_same<T, AffineType::SE3>::value) {
      return Eigen::Matrix<double, 6, 1>{0, 0, 0, 0, 0, 0};
    }
    if constexpr (std::is_same<T, AffineType::Sim3>::value) {
      return Eigen::Matrix<double, 7, 1>{0, 0, 0, 0, 0, 0, 0};
    }
  }

protected:
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
};

template <typename T2>
[[maybe_unused]] static Sim3_t operator*(const Sim3_t& lhs,
                                         const LieGroup_t<T2>& rhs) {
  return Sim3_t{lhs.R_ * rhs.R_, lhs.R_ * rhs.t() + lhs.t()};
}
template <typename T1>
[[maybe_unused]] static Sim3_t operator*(const LieGroup_t<T1>& lhs,
                                         const Sim3_t& rhs) {
  return {lhs.R() * rhs.R_, lhs.R() * rhs.t() + lhs.t()};
}

[[maybe_unused]] static SE3_t operator*(const SE3_t& lhs, const SO3_t& rhs) {
  return {lhs.R() * rhs.R(), lhs.t()};
}
[[maybe_unused]] static SE3_t operator*(const SO3_t& lhs, const SE3_t& rhs) {
  return {lhs.R() * rhs.R(), lhs.R() * rhs.t()};
}

[[maybe_unused]] static SE3_t to_SE3(const Sim3_t& sim3) {
  const double inv_scale = 1.0 / sim3.s();
  return {sim3.R(), inv_scale * sim3.t()};
}

} // namespace lie

#endif // LIE_TYPES_H
