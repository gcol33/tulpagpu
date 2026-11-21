#ifndef TULPAGPU_ROUTES_H
#define TULPAGPU_ROUTES_H

#include "context.h"

namespace tulpagpu {
using tulpa::device::BatchView;
using tulpa::device::DenseBasisRoute;
using tulpa::device::DenseFactorRoute;
using tulpa::device::DenseScatterRoute;
using tulpa::device::DenseSystem;
using tulpa::device::LikelihoodRoute;
using tulpa::device::LikelihoodResult;
using tulpa::device::ObservationPlan;
using tulpa::device::SparseFactorRoute;

std::unique_ptr<DenseBasisRoute> make_dense_basis();
std::unique_ptr<DenseFactorRoute> make_dense_factor();
std::unique_ptr<DenseScatterRoute> make_dense_scatter();
std::unique_ptr<SparseFactorRoute> make_sparse_factor();
std::unique_ptr<LikelihoodRoute> make_likelihood(const ObservationPlan& plan);

Status cross_product(const double* left, int rows, int columns, const double* right,
                     int right_columns, double* out);
Status cholesky_adjoint(const double* lower, const double* adjoint, double* out, int n);
Status cholesky_batch(double* matrices, int n, int count);
Status axpy(const double* x, double* y, double a, int n);

}
#endif
