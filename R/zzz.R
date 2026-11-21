.onLoad <- function(libname, pkgname) {
  cpp_gpu_register(.device_thresholds(.device_timings(libname, pkgname)),
                   .device_kernel_sources(libname, pkgname))
}

# The DLL stays loaded: device routes tulpa already holds run its code.
.onUnload <- function(libpath) {
  cpp_gpu_unregister()
}
