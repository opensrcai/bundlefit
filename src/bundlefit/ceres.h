#ifndef BUNDLEFIT_CERES_H
#define BUNDLEFIT_CERES_H

#include <ceres/cost_function.h>
#include <ceres/loss_function.h>

#include <Eigen/Core>

namespace bundlefit {
// Compute cost and if the residual is inlier
static std::pair<double, bool> compute_cost(
    const ceres::CostFunction* cost_function,
    const ceres::LossFunction* loss_function,
    const std::vector<double*>& param_block, const uint32_t num_residuals,
    double** jacobians = nullptr) {
  double** params   = new double*[param_block.size()];
  double* residuals = new double[num_residuals];
  for (size_t i = 0; i < param_block.size(); i++) {
    params[i] = param_block.at(i);
  }
  cost_function->Evaluate(params, residuals, jacobians);
  const double sq_norm
      = Eigen::Map<Eigen::VectorXd>(residuals, num_residuals).squaredNorm();
  delete[] params;
  delete[] residuals;

  if (loss_function) {
    double rho[3];
    loss_function->Evaluate(sq_norm, rho);
    return {0.5 * rho[0], rho[1] >= 1.0};
  }
  else {
    return {0.5 * sq_norm, true};
  }
}
} // namespace bundlefit

#endif // BUNDLEFIT_CERES_H
