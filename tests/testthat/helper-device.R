skip_if_no_cuda <- function(capability = "driver") {
  info <- gpu_info()
  if (!isTRUE(info$available)) testthat::skip(paste("CUDA unavailable:", info$reason))
  for (required in capability) {
    reason <- info$library_reasons[[required]]
    if (is.null(reason)) stop("unknown CUDA capability")
    if (nzchar(reason)) testthat::skip(paste("CUDA", required, "unavailable:", reason))
  }
  invisible(info)
}

skip_if_not_slow <- function() {
  if (!identical(tolower(Sys.getenv("TULPA_SLOW_TESTS")), "true")) {
    testthat::skip("full-validation tier: set TULPA_SLOW_TESTS=true to run the large device fits")
  }
}

device_nngp_fixture <- function(batch, nn) {
  n <- as.integer(batch + 1L)
  i <- seq_len(n)
  coords <- cbind(i * 0.02, sin(i * 0.73) * 0.1)
  idx <- matrix(0L, n, nn)
  distances <- matrix(0, n, nn)
  for (k in seq_len(nn)) {
    rows <- seq.int(k + 1L, n)
    idx[rows, k] <- rows - k
    distances[rows, k] <- sqrt(rowSums((coords[rows, , drop = FALSE] -
                                       coords[rows - k, , drop = FALSE])^2))
  }
  list(w = sin(i * 0.33), coords = coords, nn_idx = idx, nn_dist = distances,
       nn_order = seq.int(0L, n - 1L), sigma2 = 0.9, phi = 0.4, cov_type = 0L)
}

smc_nngp_fixture <- function(name) {
  cfg <- switch(name,
    F1 = c(J = 12, n = 72, s2 = 0.5, range = 0.3, noise = 0.25, seed = 20))
  set.seed(cfg[["seed"]])
  J <- as.integer(cfg[["J"]]); n <- as.integer(cfg[["n"]])
  xy <- data.frame(lon = runif(J), lat = runif(J))
  loc <- rep_len(1:J, n); x <- rnorm(n)
  d <- data.frame(lon = xy$lon[loc], lat = xy$lat[loc], x = x)
  sp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(
    spatial_gp(~lon + lat, nn = J - 1L), d))
  D <- as.matrix(dist(sp$coords))
  A <- outer(sp$obs_to_loc + 1L, 1:J, "==") * 1
  w <- t(chol(cfg[["s2"]] * exp(-D / cfg[["range"]]) + diag(1e-8, J))) %*% rnorm(J)
  y <- 1 + 0.5 * x + as.numeric(A %*% w) + rnorm(n, 0, sqrt(cfg[["noise"]]))
  stopifnot(length(y) == n, nrow(sp$coords) == J, ncol(A) == J)
  list(y = y, X = cbind(1, x), A = A, D = D, sp = sp, noise = cfg[["noise"]])
}

expect_device_equal <- function(actual, expected, tolerance, quantity) {
  testthat::expect_identical(length(actual), length(expected), info = quantity)
  testthat::expect_identical(dim(actual), dim(expected), info = quantity)
  if (length(actual) != length(expected) || !identical(dim(actual), dim(expected)))
    return(invisible(NULL))
  values <- as.numeric(actual); reference <- as.numeric(expected)
  finite <- is.finite(reference)
  testthat::expect_identical(is.finite(values), finite, info = quantity)
  if (!identical(is.finite(values), finite)) return(invisible(NULL))
  testthat::expect_identical(values[!finite], reference[!finite], info = quantity)
  scale <- max(abs(reference[finite]), 1)
  error <- max(c(0, abs(values[finite] - reference[finite])))
  output <- Sys.getenv("TULPA_DEVICE_EQUIVALENCE", "")
  if (nzchar(output)) {
    row <- data.frame(quantity = quantity, scale = scale,
      absolute_error = error, relative_tolerance = tolerance)
    utils::write.table(row, output, sep = ",", append = file.exists(output),
      row.names = FALSE, col.names = !file.exists(output), quote = TRUE)
  }
  testthat::expect_true(error <= tolerance * scale,
    info = sprintf("%s: reference scale %.17g, relative tolerance %.3g",
                   quantity, scale, tolerance))
}
