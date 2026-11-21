test_that('every device operation has named equivalence evidence and benchmark rows', {
  map <- device_equivalence_map()
  expect_setequal(map$operation,tulpa:::cpp_device_operations())
  timings <- read.csv(system.file('device','benchmarks.csv',package='tulpaGPU'))
  expect_true(all(map$operation %in% timings$operation))
  for (file in unique(map$file)) {
    titles <- device_test_names(testthat::test_path(file))
    expect_true(all(map$test[map$file==file] %in% titles),info=file)
  }
})
