#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <vector>

#include "bundlefit/bundle_adjuster.h"
#include "bundlefit/types.h"
#include "lie/lie_3d.h"

using namespace bundlefit;

// std::vector for Eigen with allocator
template <typename T>
using eigen_std_vector = std::vector<T, Eigen::aligned_allocator<T>>;

std::mt19937 mt;

const double ERROR_THRES = 0.4;

// Create vector from array
template <int N>
Vec_t<N> MakeVec(const double (&array)[N]) {
  Vec_t<N> vec;
  for (uint32_t i = 0; i < N; i++) {
    vec(i) = array[i];
  }
  return vec;
}

eigen_std_vector<Eigen::Vector3d> gen_corridor(const int num_W, const int num_H,
                                               const int num_D,
                                               const double L) {
  eigen_std_vector<Eigen::Vector3d> points;
  const double a = L / (num_W - 1);
  // Repeat toward front
  for (int d = 0; d < num_D; d++) {
    const double z  = d * a;
    const double y0 = 0;

    // Ground plane
    for (int i = 0; i < num_W; i++) {
      const double x = (i - (num_W - 1) / 2.0) * a;
      points.emplace_back(Eigen::Vector3d{x, y0, z});
    }
    // Side walls
    for (int i = 1; i < num_H; i++) {
      const double x = (num_W - 1) / 2.0 * a;
      const double y = -i * a;
      points.emplace_back(Eigen::Vector3d{-x, y, z});
      points.emplace_back(Eigen::Vector3d{x, y, z});
    }
  }
  return points;
}

Eigen::Vector3d gen_random_3D(const Eigen::Vector3d& center, const double sd) {
  std::normal_distribution<double> dist(0, sd);
  return center + Eigen::Vector3d{dist(mt), dist(mt), dist(mt)};
}

std::vector<cv::Mat> visualize(const BundleAdjuster& bundle_adjuster,
                               const eigen_std_vector<Eigen::Vector3d>& points,
                               const double focal, const int W, const int H,
                               const Eigen::Vector3d& center) {
  const double motion_r   = 0.5;
  const size_t num_points = points.size();

  std::vector<cv::Mat> animation;
  for (int j = 0; j < 30; j++) {
    cv::Mat vis(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    const auto theta = j / 30.0 * 2 * M_PI;
    const Eigen::Vector3d cam_center
        = center
          + Eigen::Vector3d{motion_r * cos(theta), motion_r * sin(theta), 0};

    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t = -R * cam_center;

    for (size_t i = 0; i < num_points; i++) {
      const auto gt_point         = points.at(i);
      const auto point            = bundle_adjuster.get_landmark_position(i);
      const Eigen::Vector3d pos_l = R * point + t;
      const double x              = pos_l(0);
      const double y              = pos_l(1);
      const double z              = pos_l(2);
      const double u              = x / z * focal + W / 2;
      const double v              = y / z * focal + H / 2;
      cv::Scalar color(0, 192, 0);
      if ((point - gt_point).norm() > ERROR_THRES) {
        color = cv::Scalar(0, 0, 255);
      }
      cv::circle(vis, cv::Point2f(u, v), 2, color, 1);
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
  // Generate corridor-shaped points
  // width 4[m], height: 4[m], depth: 20[m]
  const auto points     = gen_corridor(10, 10, 50, 4);
  const auto num_points = points.size();

  const size_t num_shots = 10;

  const double point_init_noise = 1;
  const double rot_init_noise   = 0.1;
  const double trans_init_noise = 1.0;

  eigen_std_vector<Eigen::Matrix3d> Rs;
  eigen_std_vector<Eigen::Vector3d> ts;
  for (size_t i = 0; i < num_shots; i++) {
    // Center is {0, -2, Z}, Z = 0 ... 9
    const Eigen::Vector3d trj = gen_random_3D({0, -2, i - 6.2}, 0.1);
    const Eigen::Vector3d r   = gen_random_3D({0, 0, 0}, 0.03);
    const Eigen::Matrix3d R   = lie::SO3::exp(r);
    const Eigen::Vector3d t   = -R * trj;

    Rs.push_back(R);
    ts.push_back(t);
  }

  const double focal = 800;
  const int W        = 1280;
  const int H        = 720;

  // 1. Construct sparse bundle adjuster
  BundleAdjuster bundle_adjuster(1, true);

  // 2. Add perspective camera into bundle_adjuster
  bundle_adjuster.add_perspective_camera(0, focal, focal, W / 2, H / 2, 0,
                                         FLAG_FIX_INTRINSIC_PARAMS);

  // 3. Add shot
  {
    // First shot is fixed in true position
    const Eigen::Matrix3d& R = Rs.at(0);
    const Eigen::Vector3d& t = ts.at(0);
    bundle_adjuster.add_SE3_shot(0, lie::SE3::log(R, t), FLAG_FIX_PARAMS);
  }
  {
    // Second shot is also fixed in true position
    const Eigen::Matrix3d& R = Rs.at(1);
    const Eigen::Vector3d& t = ts.at(1);
    bundle_adjuster.add_SE3_shot(1, lie::SE3::log(R, t), FLAG_FIX_PARAMS);
  }

  for (size_t i = 2; i < num_shots; i++) {
    const Eigen::Matrix3d& R = Rs.at(i);
    const Eigen::Vector3d& t = ts.at(i);
    const Eigen::Vector3d dr = gen_random_3D({0, 0, 0}, rot_init_noise);
    const Eigen::Vector3d dt = gen_random_3D({0, 0, 0}, trans_init_noise);
    Eigen::Matrix<double, 6, 1> shot_params = lie::SE3::log(R, t);
    shot_params.topRows(3) += dr;
    shot_params.bottomRows(3) += dt;
    bundle_adjuster.add_SE3_shot(i, shot_params, false);
  }

  // 4. Add point observation
  for (size_t i = 0; i < num_points; i++) {
    const Eigen::Vector3d& point = points.at(i);
    bundle_adjuster.add_landmark(
        i, point + gen_random_3D({0, 0, 0}, point_init_noise), false);

    for (size_t j = 0; j < num_shots; j++) {
      const Eigen::Matrix3d& R = Rs.at(j);
      const Eigen::Vector3d& t = ts.at(j);

      const Eigen::Vector3d pos_l = R * point + t;
      const double x              = pos_l(0);
      const double y              = pos_l(1);
      const double z              = pos_l(2);
      const double u              = x / z * focal + W / 2;
      const double v              = y / z * focal + H / 2;
      if (z > 0 && 0 <= u && u < W && 0 <= v && v < H) {
        bundle_adjuster.add_reprojection_error(0, j, i, MakeVec({u, v}), 1.0F,
                                               LossType::TRIVIAL, 1.0);
      }
    }
  }

  const bool show_image = std::getenv("SHOW_IMAGE") != nullptr
                          && std::string(std::getenv("SHOW_IMAGE")) == "ON";

  if (show_image) {
    const auto before_anim
        = visualize(bundle_adjuster, points, focal, W, H, {0, -2, -4});
    show_animation(before_anim);
  }

  bundle_adjuster.construct_problem();

  // 5. Execute robust BA

  const auto t0 = std::chrono::steady_clock::now();
  bundle_adjuster.fit(20);
  const auto t1 = std::chrono::steady_clock::now();
  const auto elapsed
      = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();

  std::cout << "computation time: " << elapsed * 1000.0 << "[ms]" << std::endl;

  /*
  std::vector<std::pair<double, size_t>> error_list;

  for (size_t i = 0; i < num_points; i++) {
      const auto gt_point = points.at(i);
      const auto opt_point = bundle_adjuster.get_landmark_position(i);

      const auto dist = (gt_point - opt_point).norm();

      if (dist > ERROR_THRES) {
          error_list.emplace_back(dist, i);
      }
  }

  std::sort(error_list.begin(), error_list.end());

  for (const auto& [dist, i] : error_list) {
      std::cout << "landmark " << i << " have a huge error: " << dist <<
  std::endl; std::cout << "est: " <<
  bundle_adjuster.get_landmark_position(i).transpose()
  << ", gt: " << points.at(i).transpose() << std::endl; std::cout <<
  bundle_adjuster.profile_landmark(i) << std::endl; std::cout << std::endl;
  }
   */

  if (show_image) {
    const auto after_anim
        = visualize(bundle_adjuster, points, focal, W, H, {0, -2, -4});
    show_animation(after_anim);
  }

  return 0;
}
