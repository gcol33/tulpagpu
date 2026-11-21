test_that("gpu_info describes one usable context or a concrete reason", {
  info <- gpu_info()
  expect_s3_class(info, "tulpa_gpu_info")
  expect_identical(info$initializations, 1L)
  expect_named(info$libraries, c("driver", "cusolver", "cublas", "nvrtc", "cudss"))
  if (info$available) {
    expect_gt(info$device_count, 0L)
    expect_true(nzchar(info$name))
    expect_identical(info$reason, "")
  } else {
    expect_identical(info$device_count, 0L)
    expect_true(nzchar(info$reason))
    expect_output(print(info), "CUDA unavailable")
  }
})


skip_on_cran()

test_that("concurrent initialization calls share one context", {
  skip_on_cran()
  skip_if_no_cuda()
  expect_identical(cpp_gpu_test("init", numeric(), threads = 16L)$initializations, 1L)
})

test_that("RAII buffers preserve values and reject an oversized copy", {
  skip_on_cran()
  skip_if_no_cuda()
  x <- seq(-2, 2, length.out = 257)
  expect_identical(cpp_gpu_test("buffer", x)$output, x)
  expect_identical(cpp_gpu_test("buffer", numeric())$output, numeric())
  expect_error(cpp_gpu_test("bounds", x), "copy exceeds allocation")
  expect_identical(cpp_gpu_test("buffer", x)$output, x)
})

test_that("NVRTC compiles shipped source and launches a driver kernel", {
  skip_on_cran()
  skip_if_no_cuda("nvrtc")
  x <- seq(-5, 5, length.out = 513)
  y <- cos(x)
  result <- tulpa:::cpp_device_axpy(x, y, 0.13, "cuda")
  expect_true(result$used_cuda)
  expect_device_equal(result$value, y + 0.13 * x, 1e-14, "axpy")
  expect_error(cpp_gpu_test("compile", numeric(), "invalid CUDA source"),
               "NVRTC::nvrtcCompileProgram failed")
  expect_device_equal(tulpa:::cpp_device_axpy(x, y, 0.13, "cuda")$value,
                      y + 0.13 * x, 1e-14, "axpy after failed compile")
})
