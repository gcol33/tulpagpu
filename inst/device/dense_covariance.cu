#include "tulpa/dense_covariance.h"

struct TulpaDenseExp {
    double lower, upper;
    __device__ double operator()(double x) const { return exp(fmin(upper, fmax(lower, x))); }
};
extern "C" __global__ void tulpa_dense_covariance_kernel(const double* times,
    double* matrix, int n, double sigma2, double phi, int kind, double period,
    double nugget, double exp_lower, double exp_upper) {
    const unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (unsigned long long)n * n) return;
    const int i = k % n, j = k / n;
    const double d = fabs(times[i] - times[j]);
    const double s = kind == 4 ? sin(3.14159265358979323846 * d / period) : 0.0;
    matrix[k] = tulpa_dense_covariance(d, sigma2, phi, kind, s * s, TulpaDenseExp{exp_lower, exp_upper})
              + (i == j ? nugget : 0.0);
}