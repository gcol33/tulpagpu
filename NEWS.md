# tulpaGPU 0.1.0

* First release. The CUDA routes of tulpa 0.8.0 (dense and sparse Cholesky
  factorizations, batched Laplace factorizations, basis products, observation
  scatter and fused likelihood kernels), the runtime-loaded CUDA layer, the
  calibrated crossover table and `gpu_info()` move here from tulpa. The package
  registers a backend through tulpa's `device_backend.h` interface when it loads.
