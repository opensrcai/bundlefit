#include "bundlefit/pose_graph_optimizer.h"

#include <ceres/loss_function.h>
#include <ceres/solver.h>

#include "bundlefit/Sim3.h"
#include "bundlefit/autodiff_cost_functions.h"
#include "bundlefit/ceres.h"
#include "bundlefit/pose_graph_residual.h"

namespace bundlefit {

PoseGraphOptimizer::PoseGraphOptimizer(const uint32_t num_threads,
                                       const bool use_Sim3_cache)
    : problem_(new ceres::Problem())
    , num_threads_(num_threads)
    , use_Sim3_cache_(use_Sim3_cache) {}

PoseGraphOptimizer::~PoseGraphOptimizer() {
  for (auto residual : residuals_) {
    delete residual;
  }
  for (auto shot : shots_) {
    delete shot.second;
  }
  delete problem_;
}

void PoseGraphOptimizer::add_SE3_shot(const uint32_t id, const Eigen::Matrix<double, 6, 1>& params,
                                      const bool is_fixed) {
  auto shot  = new Shot(id, params, is_fixed, use_Sim3_cache_);
  shots_[id] = shot;
}

Eigen::Matrix<double, 6, 1> PoseGraphOptimizer::get_SE3_shot_pose(const uint32_t id) const {
  const auto shot = shots_.at(id);
  assert(shot->type == Shot::Type::SE3
         && "Sim3 pose acquired but the shot has SE3 pose");
  return Eigen::Map<Eigen::Matrix<double, 6, 1>>(shot->params);
}

void PoseGraphOptimizer::add_Sim3_shot(const uint32_t id, const Eigen::Matrix<double, 7, 1>& params,
                                       const bool is_fixed) {
  auto shot  = new Shot(id, params, is_fixed, use_Sim3_cache_);
  shots_[id] = shot;
}

Eigen::Matrix<double, 7, 1> PoseGraphOptimizer::get_Sim3_shot_pose(const uint32_t id) const {
  const auto shot = shots_.at(id);
  assert(shot->type == Shot::Type::Sim3
         && "SE3 pose acquired but the shot has Sim3 pose");
  return Eigen::Map<Eigen::Matrix<double, 7, 1>>(shot->params);
}

Eigen::Matrix<double, 6, 1> PoseGraphOptimizer::get_shot_pose_as_SE3(const uint32_t id) const {
  const auto shot = shots_.at(id);
  Eigen::Matrix<double, 6, 1> params;
  for (int i = 0; i < 6; i++) {
    params(i) = shot->params[i];
  }

  if (shot->type == Shot::Type::Sim3) {
    const double inv_scale = std::exp(-shot->params[6]);
    for (int i = 3; i < 6; i++) {
      params(i) *= inv_scale;
    }
  }
  return params;
}

void PoseGraphOptimizer::add_pose_graph_residual(const uint32_t shot1_id,
                                                 const uint32_t shot2_id,
                                                 const Eigen::Matrix<double, 6, 1>& pose_21) {
  if (shot1_id == shot2_id) {
    return;
  }
  const auto error = new PoseGraphResidual(shots_.at(shot1_id),
                                           shots_.at(shot2_id), pose_21);
  residuals_.push_back(error);
}

void PoseGraphOptimizer::add_pose_graph_residual(const uint32_t shot1_id,
                                                 const uint32_t shot2_id,
                                                 const Eigen::Matrix<double, 7, 1>& pose_21) {
  if (shot1_id == shot2_id) {
    return;
  }
  const auto error = new PoseGraphResidual(shots_.at(shot1_id),
                                           shots_.at(shot2_id), pose_21);
  residuals_.push_back(error);
}

void PoseGraphOptimizer::construct_problem() {
  for (auto& i : shots_) {
    auto shot = i.second;
    problem_->AddParameterBlock(shot->params,
                                int(Shot::NUM_PARAMS[shot->type]));
    if (shot->is_fixed) {
      problem_->SetParameterBlockConstant(shot->params);
    }
  }

  for (auto& residual : residuals_) {
    auto shot1 = residual->shot_1;
    auto shot2 = residual->shot_2;

    auto cost_function
        = residual->generate_cost_function(/*use_analytic_diff_*/); // TODO

    auto res_id = problem_->AddResidualBlock(cost_function, nullptr,
                                             shot1->params, shot2->params);
    residual->residual_block_id = res_id;
  }
}

void PoseGraphOptimizer::fit(uint32_t iterations, const bool dump_progress,
                             const bool dump_full_progress) {
  assert(problem_ && "construct_problem has not be called");
  ceres::Solver::Summary summary;

  ceres::Solver::Options options;
  options.linear_solver_type           = ceres::SPARSE_SCHUR;
  options.num_threads                  = int(num_threads_);
  options.max_num_iterations           = int(iterations);
  options.minimizer_progress_to_stdout = dump_progress;
  ceres::Solve(options, problem_, &summary);
  if (dump_full_progress) std::cout << summary.FullReport() << std::endl;
}

void PoseGraphOptimizer::evaluate() {
  for (const auto& res : residuals_) {
    auto cost_function
        = res->generate_cost_function(/*use_analytic_diff_*/); // TODO
    const auto parameter_block = res->stack_parameter_blocks();
    const auto& [cost, _]
        = compute_cost(cost_function, nullptr, parameter_block, res->n_dim);

    std::cout << "[" << res->shot_1->id << ", " << res->shot_2->id
              << "] : " << cost << std::endl;
  }
}

} // namespace bundlefit
