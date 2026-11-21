#ifndef TULPAGPU_CONTEXT_H
#define TULPAGPU_CONTEXT_H

#include <tulpa/device_backend.h>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tulpagpu {
using tulpa::device::Capability;
using tulpa::device::Info;
using tulpa::device::Status;

struct Launch {
    unsigned grid_x = 1, grid_y = 1, grid_z = 1;
    unsigned block_x = 256, block_y = 1, block_z = 1;
    unsigned shared_bytes = 0;
    void* stream = nullptr;
};

// The CUDA driver and the runtime libraries, loaded on first use. CUDA types
// stay private to context.cpp: callers use integer device addresses and
// opaque handles, so building the package needs no toolkit.
class Context {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Context();
public:
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    // Never destroyed: the engine's thread caches may release device routes
    // during process exit, after this package's statics are gone.
    static Context& instance();
    Status initialize();
    Status supports(Capability capability);
    Info info();
    Status allocate(unsigned long long& pointer, std::size_t bytes);
    Status release(unsigned long long pointer);
    Status upload(unsigned long long pointer, const void* data, std::size_t bytes);
    Status download(void* data, unsigned long long pointer, std::size_t bytes);
    Status push();
    Status pop();
    Status synchronize();
    Status create_stream(void*& stream);
    Status destroy_stream(void* stream);
    Status synchronize_stream(void* stream);
    Status memory_available(std::size_t& free, std::size_t& total);
    Status gemv(unsigned long long matrix, unsigned long long x,
                unsigned long long y, int rows, int cols, bool transpose,
                double beta = 0, void* stream = nullptr);
    Status factor(unsigned long long pointers, int n, int batch,
                  unsigned long long info, void* solver = nullptr);
    Status create_solver(void*& solver, void* stream);
    Status destroy_solver(void* solver);
    Status launch(const std::string& source, const std::string& name,
                  const Launch& shape, void** arguments);
    Status compile(const std::string& source, const std::string& name,
                   void*& module, void*& kernel);
    Status unload(void* module);
    // Typed library entry points, each result checked with Status::read.
    // Handle use is protected by the same lease.
    void* symbol(Capability capability, const char* name, Status& status);
    void* handle(Capability capability, Status& status);
    std::mutex& execution_mutex();
};

class Lease {
    Context& context_;
    Status& status_;
    std::unique_lock<std::mutex> lock_;
    bool pushed_ = false;
public:
    Lease(Context& context, Status& status);
    ~Lease();
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
};

// Makes the context current without the execution lock, for work on a
// route's own stream that runs concurrently with other cells.
struct Frame {
    Context& context;
    Status& status;
    bool pushed;
    Frame(Context& c, Status& s) : context(c), status(s), pushed(false) {
        const auto result = context.push();
        pushed = bool(result); status.absorb(result);
    }
    ~Frame() { if (pushed) status.absorb(context.pop()); }
};

class Buffer {
    Context& context_;
    Status& status_;
    unsigned long long pointer_ = 0;
    std::size_t bytes_;
public:
    Buffer(Context& context, std::size_t bytes, Status& status);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    unsigned long long address() const { return pointer_; }
    std::size_t size() const { return bytes_; }
    Status upload(const void* data, std::size_t bytes);
    Status download(void* data, std::size_t bytes);
};

// Kernel sources (the inst/device .cu files and the engine's portable math
// headers) and the calibrated thresholds, both set from R at load.
void configure(const std::vector<tulpa::device::Threshold>& thresholds,
               const std::map<std::string, std::string>& sources);
void configure_thresholds(const std::vector<tulpa::device::Threshold>& thresholds);
std::vector<tulpa::device::Threshold> thresholds();
std::string kernel_source(const std::string& name);

// The engine's services, set when the backend registers.
tulpa::device::Host& host();
void set_host(tulpa::device::Host* value);

}
#endif
