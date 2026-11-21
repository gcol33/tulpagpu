# tulpaGPU

tulpaGPU runs the device operations of the [tulpa](https://gillescolling.com/tulpa/)
engine on an NVIDIA GPU. Once it is installed, tulpa loads it when a fit asks for
`control = list(device = "auto")` (the default) or `device = "cuda"`, and sends each
operation to the device when it lies above its measured size crossover:

- dense Cholesky factorizations of Laplace Newton systems, batched across the cells
  of a nested Laplace grid
- sparse Cholesky factorizations and selected inverses through cuDSS
- dense Gaussian process covariance factors, basis products and projected
  conditionals
- fused likelihood value, score and curvature kernels for NUTS and spec-driven
  Laplace

Without tulpaGPU, tulpa runs every operation on the CPU.

## Installation

```r
install.packages("tulpaGPU", repos = c("https://gcol33.r-universe.dev", getOption("repos")))
```

Installing needs a C++ compiler only. At run time the package loads the NVIDIA
driver and the CUDA runtime libraries (cuSOLVER, cuBLAS, NVRTC, and cuDSS for the
sparse routes) from the system; on Windows it also looks under `CUDA_PATH` and
`CUDSS_DIR`.

## Usage

```r
library(tulpaGPU)
gpu_info()
#> CUDA: NVIDIA GeForce RTX 5080, 16275 MiB, compute capability 12.0

fit <- tulpa::tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                    control = list(device = "cuda"))
fit$device$stages
```

`fit$device$stages` records the CPU and CUDA calls of every stage of the fit.
`device = "cuda"` sends every registered operation to the device and stops with the
reason when the device or a library is unavailable.
