#include "context.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#define TULPA_DEVICE_CALL __stdcall
#else
#include <dlfcn.h>
#define TULPA_DEVICE_CALL
#endif

namespace tulpagpu {
namespace detail {
std::mutex settings_mutex;
std::vector<tulpa::device::Threshold> settings;
std::map<std::string, std::string> kernel_sources;
std::size_t source_generation=0;

class Library {
    void* handle_ = nullptr;
public:
    std::string name, reason;
    ~Library() {
        if (!handle_) return;
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
    }
    bool open(const std::vector<std::string>& names) {
#ifndef TULPA_DISABLE_CUDA
        for (const auto& candidate : names) {
#ifdef _WIN32
            handle_ = LoadLibraryExA(candidate.c_str(), nullptr,
                candidate.find('/') == std::string::npos ? 0 : LOAD_WITH_ALTERED_SEARCH_PATH);
#else
            handle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            if (handle_) { name = candidate; return true; }
        }
#endif
        reason = "cannot load " + names.front();
        return false;
    }
    void* find(const char* symbol, Status& status) {
        if (!status) return nullptr;
#ifdef _WIN32
        void* result = handle_ ? reinterpret_cast<void*>(
            GetProcAddress(static_cast<HMODULE>(handle_), symbol)) : nullptr;
#else
        void* result = handle_ ? dlsym(handle_, symbol) : nullptr;
#endif
        if (!result) status = Status::read("loader", -1, symbol,
            handle_ ? "entry point absent from " + name : reason);
        return result;
    }
    bool loaded() const { return handle_ != nullptr; }
};

std::vector<std::string> library_names(const char* windows, const char* posix) {
#ifdef _WIN32
    std::vector<std::string> result{windows};
    for (const char* variable : {"CUDA_PATH", "CUDSS_DIR"}) {
        const char* root = std::getenv(variable);
        if (!root || !*root) continue;
        for (const char* suffix : {"/bin/x64/", "/bin/"})
            result.push_back(std::string(root) + suffix + windows);
    }
    return result;
#else
    return {posix};
#endif
}

using Address = unsigned long long;
using Handle = void*;
#define DEVICE_FN(name, args) using name##_type = int (TULPA_DEVICE_CALL*) args; name##_type name = nullptr
}
using namespace detail;

struct Context::Impl {
    std::once_flag once;
    std::mutex execution;
    Status status;
    Status capability[5];
    Library libraries[5];
    Handle context = nullptr, blas = nullptr, solver = nullptr;
    int device = 0;
    struct CompiledKernel { Handle module = nullptr, function = nullptr; };
    std::map<std::tuple<std::size_t,std::string,std::string>, CompiledKernel> kernels;
    Info info;
    DEVICE_FN(init, (unsigned));
    DEVICE_FN(driver_version, (int*));
    DEVICE_FN(device_count, (int*));
    DEVICE_FN(device_get, (int*, int));
    DEVICE_FN(device_name, (char*, int, int));
    DEVICE_FN(device_memory, (std::size_t*, int));
    DEVICE_FN(device_attribute, (int*, int, int));
    DEVICE_FN(retain, (Handle*, int));
    DEVICE_FN(release, (int));
    DEVICE_FN(push, (Handle));
    DEVICE_FN(pop, (Handle*));
    DEVICE_FN(allocate, (Address*, std::size_t));
    DEVICE_FN(free, (Address));
    DEVICE_FN(upload, (Address, const void*, std::size_t));
    DEVICE_FN(download, (void*, Address, std::size_t));
    DEVICE_FN(sync, ());
    DEVICE_FN(memory_info, (std::size_t*, std::size_t*));
    DEVICE_FN(stream_create, (Handle*, unsigned));
    DEVICE_FN(stream_destroy, (Handle));
    DEVICE_FN(stream_sync, (Handle));
    DEVICE_FN(module_load, (Handle*, const void*));
    DEVICE_FN(module_unload, (Handle));
    DEVICE_FN(function, (Handle*, Handle, const char*));
    DEVICE_FN(launch, (Handle, unsigned, unsigned, unsigned, unsigned, unsigned,
                       unsigned, unsigned, Handle, void**, void**));
    DEVICE_FN(blas_create, (Handle*));
    DEVICE_FN(blas_destroy, (Handle));
    DEVICE_FN(blas_stream, (Handle, Handle));
    DEVICE_FN(gemv, (Handle, int, int, int, const double*, const double*, int,
                     const double*, int, const double*, double*, int));
    DEVICE_FN(solver_create, (Handle*));
    DEVICE_FN(solver_destroy, (Handle));
    DEVICE_FN(solver_stream, (Handle, Handle));
    DEVICE_FN(potrf, (Handle, int, int, double**, int, int*, int));
    DEVICE_FN(program_create, (Handle*, const char*, const char*, int,
                               const char* const*, const char* const*));
    DEVICE_FN(program_compile, (Handle, int, const char* const*));
    DEVICE_FN(program_destroy, (Handle*));
    DEVICE_FN(ptx_size, (Handle, std::size_t*));
    DEVICE_FN(ptx, (Handle, char*));
    DEVICE_FN(log_size, (Handle, std::size_t*));
    DEVICE_FN(log, (Handle, char*));
    DEVICE_FN(sparse_property, (int, int*));

    template<class T> void resolve(T& target, int lib, const char* symbol, Status& s) {
        target = reinterpret_cast<T>(libraries[lib].find(symbol, s));
    }
    void initialize() {
        ++info.init_count;
#ifdef TULPA_DISABLE_CUDA
        info.compiled = false;
        status = Status::read("CUDA", -1, "initialize", "TULPA_DISABLE_CUDA is defined");
#else
        libraries[0].open(library_names("nvcuda.dll", "libcuda.so.1"));
#define RESOLVE(member, symbol) resolve(member, 0, symbol, status)
        RESOLVE(init, "cuInit");
        RESOLVE(driver_version, "cuDriverGetVersion");
        RESOLVE(device_count, "cuDeviceGetCount");
        RESOLVE(device_get, "cuDeviceGet");
        RESOLVE(device_name, "cuDeviceGetName");
        RESOLVE(device_memory, "cuDeviceTotalMem_v2");
        RESOLVE(device_attribute, "cuDeviceGetAttribute");
        RESOLVE(retain, "cuDevicePrimaryCtxRetain");
        RESOLVE(release, "cuDevicePrimaryCtxRelease_v2");
        RESOLVE(push, "cuCtxPushCurrent_v2");
        RESOLVE(pop, "cuCtxPopCurrent_v2");
        RESOLVE(allocate, "cuMemAlloc_v2");
        RESOLVE(free, "cuMemFree_v2");
        RESOLVE(upload, "cuMemcpyHtoD_v2");
        RESOLVE(download, "cuMemcpyDtoH_v2");
        RESOLVE(sync, "cuCtxSynchronize");
        RESOLVE(memory_info, "cuMemGetInfo_v2");
        RESOLVE(stream_create, "cuStreamCreate");
        RESOLVE(stream_destroy, "cuStreamDestroy_v2");
        RESOLVE(stream_sync, "cuStreamSynchronize");
        RESOLVE(module_load, "cuModuleLoadData");
        RESOLVE(module_unload, "cuModuleUnload");
        RESOLVE(function, "cuModuleGetFunction");
        RESOLVE(launch, "cuLaunchKernel");
#undef RESOLVE
        if (!status) return;
#define DRIVER(call) do { status = Status::read("CUDA", call, #call); if (!status) return; } while (false)
        DRIVER(init(0));
        DRIVER(driver_version(&info.driver_version));
        DRIVER(device_count(&info.count));
        if (!info.count) { status = Status::read("CUDA", -1, "initialize", "no CUDA devices"); return; }
        int major = 0, minor = 0;
        char name[256] = {};
        std::size_t memory = 0;
        DRIVER(device_get(&device, 0));
        DRIVER(device_name(name, sizeof(name), device));
        DRIVER(device_memory(&memory, device));
        DRIVER(device_attribute(&major, 75, device));
        DRIVER(device_attribute(&minor, 76, device));
        DRIVER(retain(&context, device));
        info.name = name;
        info.memory_mb = memory / (1024.0 * 1024.0);
        info.compute_capability = std::to_string(major) + "." + std::to_string(minor);
        DRIVER(push(context));
#undef DRIVER
        libraries[1].open(library_names("cusolver64_12.dll", "libcusolver.so.12"));
        libraries[2].open(library_names("cublas64_13.dll", "libcublas.so.13"));
        libraries[3].open(library_names("nvrtc64_130_0.dll", "libnvrtc.so.13"));
        auto sparse_names = library_names("cudss64_1.dll", "libcudss.so.1");
        auto sparse_legacy_names = library_names("cudss64_0.dll", "libcudss.so.0");
        sparse_names.insert(sparse_names.end(), sparse_legacy_names.begin(), sparse_legacy_names.end());
        libraries[4].open(sparse_names);
        resolve(solver_create, 1, "cusolverDnCreate", capability[1]);
        resolve(solver_destroy, 1, "cusolverDnDestroy", capability[1]);
        resolve(solver_stream, 1, "cusolverDnSetStream", capability[1]);
        resolve(potrf, 1, "cusolverDnDpotrfBatched", capability[1]);
        if (capability[1]) capability[1] = Status::read("cuSOLVER", solver_create(&solver), "cusolverDnCreate");
        resolve(blas_create, 2, "cublasCreate_v2", capability[2]);
        resolve(blas_destroy, 2, "cublasDestroy_v2", capability[2]);
        resolve(blas_stream, 2, "cublasSetStream_v2", capability[2]);
        resolve(gemv, 2, "cublasDgemv_v2", capability[2]);
        if (capability[2]) capability[2] = Status::read("cuBLAS", blas_create(&blas), "cublasCreate_v2");
        resolve(program_create, 3, "nvrtcCreateProgram", capability[3]);
        resolve(program_compile, 3, "nvrtcCompileProgram", capability[3]);
        resolve(program_destroy, 3, "nvrtcDestroyProgram", capability[3]);
        resolve(ptx_size, 3, "nvrtcGetPTXSize", capability[3]);
        resolve(ptx, 3, "nvrtcGetPTX", capability[3]);
        resolve(log_size, 3, "nvrtcGetProgramLogSize", capability[3]);
        resolve(log, 3, "nvrtcGetProgramLog", capability[3]);
        resolve(sparse_property, 4, "cudssGetProperty", capability[4]);
        int sparse_major = 0;
        if (capability[4]) capability[4] = Status::read("cuDSS", sparse_property(0, &sparse_major), "cudssGetProperty");
        if (capability[4] && sparse_major != 1)
            capability[4] = Status::read("cuDSS", -1, "ABI", "cuDSS 1.x is required");
        Handle previous = nullptr;
        status = Status::read("CUDA", pop(&previous), "cuCtxPopCurrent_v2");
#endif
    }
    ~Impl() {
        if (!context) return;
        Status cleanup = Status::read("CUDA", push(context), "cuCtxPushCurrent_v2");
        if (cleanup) {
            for (const auto& item : kernels)
                cleanup.absorb(Status::read("CUDA", module_unload(item.second.module), "cuModuleUnload"));
            if (solver) cleanup.absorb(Status::read("cuSOLVER", solver_destroy(solver), "cusolverDnDestroy"));
            if (blas) cleanup.absorb(Status::read("cuBLAS", blas_destroy(blas), "cublasDestroy_v2"));
            Handle previous = nullptr;
            cleanup.absorb(Status::read("CUDA", pop(&previous), "cuCtxPopCurrent_v2"));
        }
        cleanup.absorb(Status::read("CUDA", release(device), "cuDevicePrimaryCtxRelease_v2"));
    }
};
#undef DEVICE_FN

Context::Context() : impl_(new Impl) {}
Context::~Context() = default;
Status Context::initialize() {
    std::call_once(impl_->once, [this] { impl_->initialize(); });
    return impl_->status;
}
Status Context::supports(Capability capability) {
    Status s = initialize();
    if (!s) return s;
    return impl_->capability[static_cast<int>(capability)];
}
Info Context::info() {
    Status s = initialize();
    Info result = impl_->info;
    result.available = static_cast<bool>(s);
    result.reason = s.message;
    if (!s) result.count = 0;
    for (int i = 0; i < 5; ++i) {
        result.libraries.push_back(impl_->libraries[i].name);
        result.library_reasons.push_back(s ? impl_->capability[i].message : s.message);
    }
    return result;
}
std::mutex& Context::execution_mutex() { return impl_->execution; }
void* Context::symbol(Capability capability, const char* name, Status& status) {
    status.absorb(supports(capability));
    return status ? impl_->libraries[static_cast<int>(capability)].find(name, status) : nullptr;
}
void* Context::handle(Capability capability, Status& status) {
    status.absorb(supports(capability));
    return status ? (capability == Capability::Solver ? impl_->solver :
                     capability == Capability::Blas ? impl_->blas : nullptr) : nullptr;
}
Status Context::push() { Status s = initialize(); return s ? Status::read("CUDA", impl_->push(impl_->context), "cuCtxPushCurrent_v2") : s; }
Status Context::pop() { Handle previous = nullptr; return Status::read("CUDA", impl_->pop(&previous), "cuCtxPopCurrent_v2"); }
Status Context::allocate(Address& pointer, std::size_t bytes) { return Status::read("CUDA", impl_->allocate(&pointer, bytes), "cuMemAlloc_v2"); }
Status Context::release(Address pointer) { return Status::read("CUDA", impl_->free(pointer), "cuMemFree_v2"); }
Status Context::upload(Address pointer, const void* data, std::size_t bytes) {
    Status s = Status::read("CUDA", impl_->upload(pointer, data, bytes), "cuMemcpyHtoD_v2");
    // Pageable host copies may return after staging. Complete the default
    // stream transfer before a nonblocking solver stream reads the allocation.
    if (s) s.absorb(synchronize_stream(nullptr));
    return s;
}
Status Context::download(void* data, Address pointer, std::size_t bytes) { return Status::read("CUDA", impl_->download(data, pointer, bytes), "cuMemcpyDtoH_v2"); }
Status Context::synchronize() { return Status::read("CUDA", impl_->sync(), "cuCtxSynchronize"); }
Status Context::create_stream(void*& stream) { return Status::read("CUDA", impl_->stream_create(&stream, 1), "cuStreamCreate"); }
Status Context::destroy_stream(void* stream) { return Status::read("CUDA", impl_->stream_destroy(stream), "cuStreamDestroy_v2"); }
Status Context::synchronize_stream(void* stream) { return Status::read("CUDA", impl_->stream_sync(stream), "cuStreamSynchronize"); }
Status Context::memory_available(std::size_t& free, std::size_t& total) {
    return Status::read("CUDA",impl_->memory_info(&free,&total),"cuMemGetInfo_v2");
}
Status Context::gemv(Address matrix, Address x, Address y, int rows, int cols,
                     bool transpose, double beta, Handle stream) {
    Status s = supports(Capability::Blas);
    if (s) s = Status::read("cuBLAS", impl_->blas_stream(impl_->blas, stream), "cublasSetStream_v2");
    const double alpha = 1;
    if (s) s = Status::read("cuBLAS", impl_->gemv(impl_->blas, transpose ? 1 : 0,
        rows, cols, &alpha, reinterpret_cast<const double*>(matrix), rows,
        reinterpret_cast<const double*>(x), 1, &beta, reinterpret_cast<double*>(y), 1),
        "cublasDgemv_v2");
    if (impl_->blas_stream && impl_->blas)
        s.absorb(Status::read("cuBLAS", impl_->blas_stream(impl_->blas, nullptr), "cublasSetStream_v2"));
    return s;
}
Status Context::factor(Address pointers, int n, int batch, Address info, void* solver) {
    return Status::read("cuSOLVER", impl_->potrf(solver ? solver : impl_->solver, 0, n,
        reinterpret_cast<double**>(pointers), n, reinterpret_cast<int*>(info), batch), "cusolverDnDpotrfBatched");
}
Status Context::create_solver(void*& solver, void* stream) {
    auto status = supports(Capability::Solver);
    if (status) status.absorb(Status::read("cuSOLVER", impl_->solver_create(&solver), "cusolverDnCreate"));
    if (status) status.absorb(Status::read("cuSOLVER", impl_->solver_stream(solver, stream), "cusolverDnSetStream"));
    return status;
}
Status Context::destroy_solver(void* solver) {
    return solver ? Status::read("cuSOLVER", impl_->solver_destroy(solver), "cusolverDnDestroy") : Status{};
}

Lease::Lease(Context& context, Status& status) : context_(context), status_(status),
    lock_(context.execution_mutex()) {
    if (status_) { status_.absorb(context_.push()); pushed_ = static_cast<bool>(status_); }
}
Lease::~Lease() { if (pushed_) status_.absorb(context_.pop()); }
Buffer::Buffer(Context& context, std::size_t bytes, Status& status) :
    context_(context), status_(status), bytes_(bytes) {
    if (bytes && status_) status_.absorb(context_.allocate(pointer_, bytes));
}
Buffer::~Buffer() { if (pointer_) status_.absorb(context_.release(pointer_)); }
Status Buffer::upload(const void* data, std::size_t bytes) {
    if (bytes > bytes_) return Status::read("buffer", -1, "upload", "copy exceeds allocation");
    return bytes ? context_.upload(pointer_, data, bytes) : Status{};
}
Status Buffer::download(void* data, std::size_t bytes) {
    if (bytes > bytes_) return Status::read("buffer", -1, "download", "copy exceeds allocation");
    return bytes ? context_.download(data, pointer_, bytes) : Status{};
}

Status Context::compile(const std::string& source, const std::string& name,
                         Handle& module, Handle& kernel) {
    Status s = supports(Capability::Compiler);
    if (!s) return s;
    Handle program = nullptr;
    std::vector<std::string> header_names, header_text;
    {
        std::lock_guard<std::mutex> lock(settings_mutex);
        for (const auto& entry : kernel_sources) if (entry.first.find(".h") != std::string::npos) {
            header_names.push_back(entry.first); header_text.push_back(entry.second);
            if (entry.first.rfind("tulpa/",0)==0) {
                header_names.push_back(entry.first.substr(6)); header_text.push_back(entry.second);
            }
        }
    }
    std::vector<const char*> names, headers;
    for (std::size_t i=0; i<header_names.size(); ++i) {
        names.push_back(header_names[i].c_str()); headers.push_back(header_text[i].c_str());
    }
    s = Status::read("NVRTC", impl_->program_create(&program, source.c_str(), "tulpa.cu",
        static_cast<int>(names.size()), headers.data(), names.data()), "nvrtcCreateProgram");
    if (!s) return s;
    const auto destroy = [this, &s](void* value) {
        s.absorb(Status::read("NVRTC", impl_->program_destroy(&value), "nvrtcDestroyProgram"));
    };
    std::unique_ptr<void, decltype(destroy)> owned_program(program, destroy);
    // The driver's device capability determines PTX's target architecture.
    const std::string target = "--gpu-architecture=compute_" +
        impl_->info.compute_capability.substr(0, impl_->info.compute_capability.find('.')) +
        impl_->info.compute_capability.substr(impl_->info.compute_capability.find('.') + 1);
    const char* options[] = {"--std=c++17", "--fmad=false", target.c_str()};
    int code = impl_->program_compile(program, 3, options);
    if (code) {
        std::size_t length = 0;
        Status log_status = Status::read("NVRTC", impl_->log_size(program, &length), "nvrtcGetProgramLogSize");
        std::vector<char> log(length ? length : 1, 0);
        if (log_status) log_status = Status::read("NVRTC", impl_->log(program, log.data()), "nvrtcGetProgramLog");
        s = Status::read("NVRTC", code, "nvrtcCompileProgram", log_status ? log.data() : log_status.message);
    }
    std::size_t length = 0;
    if (s) s = Status::read("NVRTC", impl_->ptx_size(program, &length), "nvrtcGetPTXSize");
    std::vector<char> ptx(length);
    if (s) s = Status::read("NVRTC", impl_->ptx(program, ptx.data()), "nvrtcGetPTX");
    owned_program.reset();
    if (s) s = Status::read("CUDA", impl_->module_load(&module, ptx.data()), "cuModuleLoadData");
    if (s) s = Status::read("CUDA", impl_->function(&kernel, module, name.c_str()), "cuModuleGetFunction");
    if (!s && module) { s.absorb(unload(module)); module = nullptr; kernel = nullptr; }
    return s;
}
Status Context::unload(Handle module) { return Status::read("CUDA", impl_->module_unload(module), "cuModuleUnload"); }
Status Context::launch(const std::string& source, const std::string& name,
                        const Launch& shape, void** arguments) {
    Status s;
    std::size_t generation;
    { std::lock_guard<std::mutex> lock(settings_mutex); generation=source_generation; }
    const auto key = std::make_tuple(generation,source,name);
    auto found = impl_->kernels.find(key);
    if (found == impl_->kernels.end()) {
        Impl::CompiledKernel entry;
        s = compile(source, name, entry.module, entry.function);
        if (!s) return s;
        const auto destroy = [this, &s](void* value) { s.absorb(unload(value)); };
        std::unique_ptr<void, decltype(destroy)> owned_module(entry.module, destroy);
        found = impl_->kernels.emplace(key, entry).first;
        owned_module.release();
    }
    return Status::read("CUDA", impl_->launch(found->second.function,
        shape.grid_x, shape.grid_y, shape.grid_z,
        shape.block_x, shape.block_y, shape.block_z,
        shape.shared_bytes, shape.stream, arguments, nullptr), "cuLaunchKernel");
}
Context& Context::instance() { static Context* context = new Context; return *context; }

void configure(const std::vector<tulpa::device::Threshold>& value,
               const std::map<std::string, std::string>& sources) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    settings = value; kernel_sources = sources; ++source_generation;
}
void configure_thresholds(const std::vector<tulpa::device::Threshold>& value) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    settings = value;
}
std::vector<tulpa::device::Threshold> thresholds() { std::lock_guard<std::mutex> lock(settings_mutex); return settings; }
std::string kernel_source(const std::string& name) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    auto found = kernel_sources.find(name);
    return found == kernel_sources.end() ? "" : found->second;
}

namespace { std::atomic<tulpa::device::Host*> engine_host{nullptr}; }
tulpa::device::Host& host() {
    auto* value = engine_host.load();
    if (!value) throw std::logic_error("the tulpaGPU backend is not registered with tulpa");
    return *value;
}
void set_host(tulpa::device::Host* value) { engine_host.store(value); }

}