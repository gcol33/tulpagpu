#' CUDA device availability
#'
#' Report the CUDA driver, the selected NVIDIA device, and the runtime
#' libraries tulpa's device routes use. Installation requires a C++ compiler
#' only; the libraries are loaded when a device operation is first requested.
#'
#' @return An object of class `tulpa_gpu_info` with availability, the reason
#'   for unavailability, device properties, and library paths and reasons.
#' @examples
#' gpu_info()
#' @export
gpu_info <- function() {
  structure(cpp_gpu_info(), class = "tulpa_gpu_info")
}

#' @rdname gpu_info
#' @param x A `gpu_info()` result.
#' @param ... Ignored.
#' @export
print.tulpa_gpu_info <- function(x, ...) {
  if (isTRUE(x$available)) {
    cat(sprintf("CUDA: %s, %.0f MiB, compute capability %s\n",
                x$name, x$memory_mb, x$compute_capability))
  } else cat("CUDA unavailable: ", x$reason, "\n", sep = "")
  invisible(x)
}
