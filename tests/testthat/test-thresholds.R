test_that("automatic policy has a separate measured crossover per operation and order", {
  timings <- data.frame(operation = rep("nngp_cholesky", 6),
    variant = rep(c(10L, 15L), each = 3), size = rep(c(50, 100, 200), 2),
    cpu_seconds = 1, cuda_seconds = c(2, 0.8, 0.7, 0.8, 2, 1.2))
  settings <- .device_thresholds(timings)
  expect_identical(settings$threshold[settings$variant == 10L], 100)
  expect_identical(settings$threshold[settings$variant == 15L], Inf)
  expect_equal(nrow(.device_thresholds(timings[FALSE, ])), 0L)
  expect_error(.device_thresholds(transform(timings, cpu_seconds = 0)), "positive")
  expect_error(tulpa:::cpp_device_probe("unknown", 10, 0L, "cpu"), "unregistered operation")
  for (choice in c("auto", "cpu")) {
    expect_identical(tulpa:::cpp_device_probe("nngp_cholesky", 1e6, 987L, choice)$device, "cpu")
  }
})

test_that("the status reader turns library status into an R boundary error", {
  expect_error(cpp_gpu_test("status", numeric()),
               "CUDA::test_status failed \\(status 1\\): invalid value")
})

test_that('fit calibration requires a sustained win in factor work for every model', {
  timings <- data.frame(model=c('temporal','temporal','spde','spde'),size=c(1e5,1e6,1e7,1e8))
  timings$operation <- 'sparse_cholesky'; timings$variant <- 0L
  timings$cpu_seconds <- 1
  timings$cuda_seconds <- c(1.5,1.2,.6,.5)
  expect_identical(.device_thresholds(timings)$threshold,1e7)
  timings$cuda_seconds[4L] <- 1.1
  expect_identical(.device_thresholds(timings)$threshold,Inf)
  shared <- timings[c(3L,3L),]
  shared$model <- c('spde','areal'); shared$cuda_seconds <- c(.6,1.2)
  expect_identical(.device_thresholds(shared)$threshold,Inf)
})

test_that('automatic fit validation contributes no unmeasured crossover samples', {
  measured <- data.frame(operation='sparse_cholesky',variant=0L,size=1000,
    cpu_seconds=2,cuda_seconds=1,scope='fit',model='spde')
  validation <- measured
  validation$scope <- 'fit_auto'
  validation$cpu_seconds <- validation$cuda_seconds <- NA_real_
  expect_identical(.device_thresholds(rbind(measured,validation)),
    .device_thresholds(measured))
})

test_that("likelihood policy requires the compiler and BLAS libraries", {
  info <- gpu_info()
  reasons <- c(if (!isTRUE(info$available)) info$reason,
               info$library_reasons[c("nvrtc", "cublas")])
  reasons <- reasons[nzchar(reasons)]
  for (operation in c("likelihood", "likelihood_weights")) {
    expect_identical(tulpa:::cpp_device_probe(operation, 1e7, 0L, "cpu")$device, "cpu")
    if (length(reasons)) {
      expect_identical(tulpa:::cpp_device_probe(operation, 1e7, 0L, "auto")$device, "cpu")
      expect_error(tulpa:::cpp_device_probe(operation, 1e7, 0L, "cuda"))
    } else {
      expect_identical(tulpa:::cpp_device_probe(operation, 1e7, 0L, "cuda")$device, "cuda")
    }
  }
})

test_that("the measured plan selects CPU, streams and batches independently", {
  timings <- data.frame(operation='laplace_scatter',variant=4L,size=c(50,100,200),
    cpu_seconds=1,cuda_seconds=c(2,0.8,0.7),stream_seconds=c(0.9,0.7,0.6))
  plan <- .device_thresholds(timings)
  expect_identical(plan$threshold,50)
  expect_identical(plan$batch_threshold,Inf)
  timings$stream_seconds <- c(1.5,0.9,0.8)
  plan <- .device_thresholds(timings)
  expect_identical(plan$threshold,100)
  expect_identical(plan$batch_threshold,100)
  expect_error(.device_thresholds(transform(timings,stream_seconds=-1)),
               'positive or infinite')
})

test_that("each operation uses its latest complete calibration without mixing binaries", {
  timings <- data.frame(operation=c('laplace_dense','laplace_dense','axpy'),
    variant=0L,size=100,cpu_seconds=c(2,0.5,2),cuda_seconds=1,
    calibration=c(0L,1L,0L))
  plan <- .device_thresholds(timings)
  expect_identical(plan$threshold[plan$operation=='laplace_dense'],Inf)
  expect_identical(plan$threshold[plan$operation=='axpy'],100)
  expect_error(.device_thresholds(transform(timings,calibration=0.5)),
    'non-negative integers')
})

