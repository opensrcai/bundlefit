#include <Eigen/Geometry>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

#include "bundlefit/bundle_adjuster.h"
#include "bundlefit/types.h"
#include "lie/lie_3d.h"

using namespace bundlefit;

// std::vector for Eigen with allocator
template <typename T>
using eigen_std_vector = std::vector<T, Eigen::aligned_allocator<T>>;

std::mt19937 mt;

// Create vector from array
template <int N>
Vec_t<N> MakeVec(const double (&array)[N]) {
  Vec_t<N> vec;
  for (uint32_t i = 0; i < N; i++) {
    vec(i) = array[i];
  }
  return vec;
}

eigen_std_vector<Eigen::Vector3d> gen_sphere(const double r, const double num_x,
                                             const double num_y) {
  eigen_std_vector<Eigen::Vector3d> points;
  for (int i = 0; i < num_x; i++) {
    const double theta = M_PI * 2 * i / num_x;
    for (int j = 0; j < num_y; j++) {
      const double phi = M_PI * ((j + 1) / (num_y + 1) - 0.5);
      const double x   = r * std::sin(theta) * std::cos(phi);
      const double y   = r * std::sin(phi);
      const double z   = r * std::cos(theta) * std::cos(phi);
      points.emplace_back(Eigen::Vector3d{x, y, z});
    }
  }
  return points;
}

Eigen::Vector3d gen_random_3D(const Eigen::Vector3d& center, const double sd) {
  std::normal_distribution<double> dist(0, sd);
  return center + Eigen::Vector3d{dist(mt), dist(mt), dist(mt)};
}

std::vector<cv::Mat> visualize(const eigen_std_vector<Eigen::Vector3d>& points1,
                               const Eigen::Matrix3d& R1,
                               const Eigen::Vector3d& t1, const double s1,
                               const eigen_std_vector<Eigen::Vector3d>& points2,
                               const Eigen::Matrix3d& R2,
                               const Eigen::Vector3d& t2, const double s2,
                               const double focal, const int W, const int H) {
  const double motion_r   = 0.5;
  const size_t num_points = points1.size();

  std::vector<cv::Mat> animation;
  for (int j = 0; j < 30; j++) {
    cv::Mat vis(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    const auto theta = j / 30.0 * 2 * M_PI;
    const Eigen::Vector3d cam_center{motion_r * cos(theta),
                                     motion_r * sin(theta), -20};

    const Eigen::Matrix3d Rc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d tc = -Rc * cam_center;

    for (size_t i = 0; i < num_points; i++) {
      {
        const auto point1           = points1.at(i);
        const Eigen::Vector3d pos_w = 1.0 / s1 * R1.transpose() * (point1 - t1);
        const Eigen::Vector3d pos_l = Rc * pos_w + tc;
        const double x              = pos_l(0);
        const double y              = pos_l(1);
        const double z              = pos_l(2);
        const double u              = x / z * focal + W / 2;
        const double v              = y / z * focal + H / 2;
        cv::circle(vis, cv::Point2f(u, v), 2, cv::Scalar(0, 0, 255), 1);
      }

      {
        const auto point2           = points2.at(i);
        const Eigen::Vector3d pos_w = 1.0 / s2 * R2.transpose() * (point2 - t2);
        const Eigen::Vector3d pos_l = Rc * pos_w + tc;
        const double x              = pos_l(0);
        const double y              = pos_l(1);
        const double z              = pos_l(2);
        const double u              = x / z * focal + W / 2;
        const double v              = y / z * focal + H / 2;
        cv::circle(vis, cv::Point2f(u, v), 4, cv::Scalar(0, 255, 0), 1);
      }
    }
    animation.emplace_back(std::move(vis));
  }
  return animation;
}

void show_animation(const std::vector<cv::Mat>& animation) {
  const auto num_frame = animation.size();
  for (int i = 0;; i = (i + 1) % int(num_frame)) {
    cv::imshow("animation", animation.at(i));
    const auto key = cv::waitKey(33);
    if (key == 113) {
      break;
    }
  }
}

int main() {
  const auto gt_points  = gen_sphere(4, 8, 5);
  const auto num_points = gt_points.size();

  eigen_std_vector<Eigen::Vector3d> points1;
  eigen_std_vector<Eigen::Vector3d> points2;

  const double Sim3_s = 0.9;
  const Eigen::Matrix3d Sim3_R
      = lie::SO3::exp(Eigen::Vector3d{0.1, -0.05, 0.13});
  const Eigen::Vector3d Sim3_t = Eigen::Vector3d{-2.2, 0.8, 3.1};

  Eigen::Matrix3d R1;
  Eigen::Vector3d t1;
  Eigen::Matrix3d R2;
  Eigen::Vector3d t2;
  double s2;
  {
    const Eigen::Vector3d trj = gen_random_3D({0, 0, -7}, 0.1);
    const Eigen::Vector3d r   = gen_random_3D({0, 0, 0}, 0.03);
    R1                        = lie::SO3::exp(r);
    t1                        = -R1 * trj;
  }
  {
    s2 = 1 / Sim3_s;
    R2 = Sim3_R.transpose() * R1;
    t2 = s2 * Sim3_R.transpose() * (t1 - Sim3_t);
  }

  for (const Eigen::Vector3d& point : gt_points) {
    points1.emplace_back(R1 * point + t1);
    points2.emplace_back(s2 * R2 * point + t2);
  }

  eigen_std_vector<Eigen::Matrix3d> R_offsets;
  eigen_std_vector<Eigen::Vector3d> t_offsets;
  for (uint32_t i = 0; i < 3; i++) {
    const Eigen::Vector3d trj = gen_random_3D({0, 0, -1}, 0.1) * i;
    const Eigen::Vector3d r   = gen_random_3D({0, 0, 0}, 0.03) * i;
    const Eigen::Matrix3d R   = lie::SO3::exp(r);
    R_offsets.push_back(R);
    t_offsets.push_back(-R.inverse() * trj);
  }

  const double focal = 800;
  const int W        = 1280;
  const int H        = 720;

  // 1. Construct sparse bundle adjuster
  BundleAdjuster bundle_adjuster(2, true);

  // 2. Add perspective camera into bundle_adjuster
  bundle_adjuster.add_perspective_camera(
      0, focal, focal, W / 2, H / 2,
      lie::SE3::log(R_offsets.at(0), t_offsets.at(0)),
      FLAG_FIX_INTRINSIC_PARAMS | FLAG_FIX_EXTRINSIC_PARAMS);
  bundle_adjuster.add_perspective_camera(
      1, focal, focal, W / 2, H / 2,
      lie::SE3::log(R_offsets.at(1), t_offsets.at(1)),
      FLAG_FIX_INTRINSIC_PARAMS | FLAG_FIX_EXTRINSIC_PARAMS);
  bundle_adjuster.add_perspective_camera(
      2, focal, focal, W / 2, H / 2,
      lie::SE3::log(R_offsets.at(2), t_offsets.at(2)),
      FLAG_FIX_INTRINSIC_PARAMS | FLAG_FIX_EXTRINSIC_PARAMS);

  // TODO: add initial noise
  bundle_adjuster.add_SE3_shot(0, lie::SE3::log(R1, t1), FLAG_FIX_PARAMS);
  bundle_adjuster.add_Sim3_shot(1, lie::Sim3::log(R2, t2, s2),
                                FLAG_OPTIMIZE_PARAMS);

  for (size_t i = 0; i < num_points; i++) {
    const Eigen::Vector3d& point1 = points1.at(i);
    const Eigen::Vector3d& point2 = points2.at(i);

    bundle_adjuster.add_landmark(i, point1, true, 0);
    bundle_adjuster.add_landmark(i + num_points, point2, true, 1);

    {
      const Eigen::Vector3d pos_l = point2;
      for (int j = 0; j < 3; j++) {
        Eigen::Matrix3d R_offset = R_offsets.at(j);
        Eigen::Vector3d t_offset = t_offsets.at(j);

        const Eigen::Vector3d pos_c = R_offset * pos_l + t_offset;

        const double x = pos_c(0);
        const double y = pos_c(1);
        const double z = pos_c(2);
        const double u = x / z * focal + W / 2;
        const double v = y / z * focal + H / 2;
        if (z > 0 && 0 <= u && u < W && 0 <= v && v < H) {
          bundle_adjuster.add_reprojection_error(j, 1, i, MakeVec({u, v}), 1.0F,
                                                 LossType::TRIVIAL, 1.0);
        }
      }
    }
  }

  // const auto before_anim = visualize(points1, R1, t1, 1.0, points2, R2,
  // t2, 1.0, focal, W, H); show_animation(before_anim);

  bundle_adjuster.construct_problem();
  bundle_adjuster.fit(20, true);

  Eigen::Matrix3d R2_opt;
  Eigen::Vector3d t2_opt;
  double s2_opt;
  {
    const Eigen::Matrix<double, 7, 1>& shot2_params
        = bundle_adjuster.get_Sim3_shot_params(1);
    R2_opt = lie::SO3::exp(shot2_params.topRows(3));
    t2_opt = shot2_params.middleRows(3, 3);
    s2_opt = shot2_params(6);
  }

  const auto after_anim = visualize(points1, R1, t1, 1.0, points2, R2_opt,
                                    t2_opt, s2_opt, focal, W, H);
  show_animation(after_anim);
}
