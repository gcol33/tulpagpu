#include "routes.h"
#include <tulpa/device_math.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <thread>

#ifdef _WIN32
#define TULPA_DENSE_CALL __stdcall
#else
#define TULPA_DENSE_CALL
#endif

namespace tulpagpu {
namespace {
using Address = unsigned long long;

struct Memory {
    Context& context = Context::instance();
    Status status;
    std::map<int, std::unique_ptr<Buffer>> buffers;
    ~Memory() {
        if (buffers.empty()) return;
        Status cleanup;
        Lease lease(context, cleanup);
        buffers.clear();
    }
    Buffer& reserve(int slot, std::size_t bytes) {
        auto& buffer = buffers[slot];
        if (!buffer || buffer->size() < bytes) {
            buffer.reset();
            buffer.reset(new Buffer(context, bytes, status));
        }
        return *buffer;
    }
    template<class Fn> Fn symbol(Capability capability, const char* name) {
        return reinterpret_cast<Fn>(context.symbol(capability, name, status));
    }
};

Status gemm(Memory& memory, bool ta, bool tb, int rows, int columns, int inner,
            Address a, int lda, Address b, int ldb, Address out) {
    using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, int, int,
        const double*, const double*, int, const double*, int, const double*, double*, int);
    auto fn = memory.symbol<Fn>(Capability::Blas, "cublasDgemm_v2");
    void* handle = memory.context.handle(Capability::Blas, memory.status);
    const double one = 1, zero = 0;
    if (!memory.status) return memory.status;
    return Status::read("cuBLAS", fn(handle, ta ? 1 : 0, tb ? 1 : 0,
        rows, columns, inner, &one, reinterpret_cast<double*>(a), lda,
        reinterpret_cast<double*>(b), ldb, &zero, reinterpret_cast<double*>(out), rows), "cublasDgemm_v2");
}

Status triangular_solve(Memory& memory, bool right, bool transpose, int rows, int columns,
                        Address lower, int leading_dimension, Address rhs) {
    using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, int, int, int,
        const double*, const double*, int, double*, int);
    auto fn = memory.symbol<Fn>(Capability::Blas, "cublasDtrsm_v2");
    auto handle = memory.context.handle(Capability::Blas, memory.status);
    const double one = 1;
    if (!memory.status) return memory.status;
    return Status::read("cuBLAS", fn(handle, right ? 1 : 0, 0, transpose ? 1 : 0, 0,
        rows, columns, &one, reinterpret_cast<double*>(lower), leading_dimension,
        reinterpret_cast<double*>(rhs), rows), "cublasDtrsm_v2");
}

struct ThreadMemory { Memory cross, adjoint; };
ThreadMemory& thread_memory() {
    static thread_local ThreadMemory* current = nullptr;
    if (current) return *current;
    Context::instance();
    struct Owned {
        std::mutex mutex;
        std::map<std::thread::id, std::unique_ptr<ThreadMemory>> threads;
    };
    static Owned owned;
    std::lock_guard<std::mutex> lock(owned.mutex);
    auto& work = owned.threads[std::this_thread::get_id()];
    if (!work) work.reset(new ThreadMemory);
    current = work.get();
    return *current;
}

// ---- batched Laplace factorization ------------------------------------------

struct FactorWorkspace {
    Context& context = Context::instance();
    Status status;
    std::unique_ptr<Buffer> pointers, info;
    std::vector<unsigned long long> addresses;
    std::vector<int> codes;
    std::size_t capacity = 0;
    void reserve(std::size_t count) {
        if (count > capacity) {
            pointers.reset(new Buffer(context, count*sizeof(unsigned long long), status));
            info.reset(new Buffer(context, count*sizeof(int), status));
            capacity = count;
        }
        addresses.resize(count); codes.resize(count);
    }
    ~FactorWorkspace() {
        if (!capacity) return;
        Frame frame(context, status);
        if (frame.pushed) { pointers.reset(); info.reset(); }
    }
};

struct FactorRequest {
    unsigned long long address;
    int n;
    void* solver = nullptr;
    void* stream = nullptr;
    FactorWorkspace* workspace = nullptr;
    bool positive = false;
    Status status;
};

void factor_requests(const std::vector<FactorRequest*>& tasks, FactorWorkspace* batch_workspace = nullptr) {
    if (tasks.empty()) return;
    auto& work = tasks.size() == 1 ? *tasks[0]->workspace : *batch_workspace;
    auto& s = work.status;
    auto& context = Context::instance();
    auto run = [&] {
        work.reserve(tasks.size());
        for (std::size_t k = 0; k < tasks.size(); ++k) work.addresses[k] = tasks[k]->address;
        if (s) s.absorb(work.pointers->upload(work.addresses.data(), tasks.size()*sizeof(unsigned long long)));
        if (s) s.absorb(context.factor(work.pointers->address(), tasks[0]->n, static_cast<int>(tasks.size()),
            work.info->address(), tasks.size() == 1 ? tasks[0]->solver : nullptr));
        if (s) s.absorb(context.synchronize_stream(tasks.size() == 1 ? tasks[0]->stream : nullptr));
        if (s) s.absorb(work.info->download(work.codes.data(), tasks.size()*sizeof(int)));
    };
    if (tasks.size() > 1) { Lease lease(context, s); if (s) run(); }
    else { Frame frame(context, s); if (s) run(); }
    for (std::size_t k = 0; k < tasks.size(); ++k) {
        tasks[k]->status = s;
        tasks[k]->positive = bool(s) && work.codes[k] == 0;
        if (s && work.codes[k] < 0)
            tasks[k]->status = Status::read("cuSOLVER", work.codes[k], "potrf batch argument", std::to_string(k));
    }
    auto policy = host().policy();
    policy->record("laplace_dense_launch", true);
    if (tasks.size() > 1) policy->record("laplace_dense_batched_launch", true);
    else policy->record("laplace_dense_stream_launch", true);
}

// One cell's factorization on its own solver and stream, joining the batched
// factorization of the calling thread's batch when the policy selects it.
Status laplace_factor(Address address, int n, bool& positive, void* solver, void* stream,
                      FactorWorkspace& own, const std::string& operation) {
    FactorRequest task{address, n, solver, stream, &own};
    auto* batch = host().batch();
    const int width = batch ? batch->width() : 1;
    if (width == 1 || !host().policy()->batch_selected(operation, n, width)) {
        factor_requests({&task});
        positive = task.positive;
        return task.status;
    }
    auto shared = std::static_pointer_cast<FactorWorkspace>(batch->claim("laplace_dense_factor",
        []() -> std::shared_ptr<void> { return std::make_shared<FactorWorkspace>(); },
        [](const std::shared_ptr<void>&) { return true; }));
    if (!shared) return Status::read("CUDA", -1, "batch", "no shared factor workspace");
    const auto key = reinterpret_cast<void*>(static_cast<std::uintptr_t>(n)*2 + 1);
    Status status = batch->submit(key, &task, [shared](const std::vector<void*>& items) {
        std::vector<FactorRequest*> tasks;
        for (auto* p : items) tasks.push_back(static_cast<FactorRequest*>(p));
        factor_requests(tasks, shared.get());
    });
    positive = task.positive;
    status.absorb(task.status);
    return status;
}

// ---- dense basis ------------------------------------------------------------

class CudaDenseBasis final : public DenseBasisRoute {
public:
    Memory memory;
    int rows = 0, columns = 0;
    Address product() const { return memory.buffers.at(7)->address(); }

    Status upload(const double* basis, int r, int c) override {
        rows = r; columns = c;
        Lease lease(memory.context, memory.status);
        auto& target = memory.reserve(0, sizeof(double) * static_cast<std::size_t>(rows) * columns);
        if (memory.status) memory.status = target.upload(basis, sizeof(double) * static_cast<std::size_t>(rows) * columns);
        return memory.status;
    }
    Status apply(const double* coefficients, double* out, bool transpose) override {
        auto& m = memory;
        const int n_in = transpose ? rows : columns, n_out = transpose ? columns : rows;
        Lease lease(m.context, m.status);
        auto& x = m.reserve(1, sizeof(double) * n_in);
        auto& y = m.reserve(2, sizeof(double) * n_out);
        if (m.status) m.status = x.upload(coefficients, sizeof(double) * n_in);
        using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, const double*, const double*, int,
            const double*, int, const double*, double*, int);
        auto fn = m.symbol<Fn>(Capability::Blas, "cublasDgemv_v2");
        auto handle = m.context.handle(Capability::Blas, m.status);
        const double one = 1, zero = 0;
        if (m.status) m.status = Status::read("cuBLAS", fn(handle, transpose ? 1 : 0,
            rows, columns, &one, reinterpret_cast<double*>(m.buffers.at(0)->address()), rows,
            reinterpret_cast<double*>(x.address()), 1, &zero, reinterpret_cast<double*>(y.address()), 1), "cublasDgemv_v2");
        if (m.status) m.status = y.download(out, sizeof(double) * n_out);
        return m.status;
    }
    Status gram(const double* weights, const double* scale, double* out) override {
        auto& m = memory;
        const std::size_t size = static_cast<std::size_t>(rows) * columns;
        Lease lease(m.context, m.status);
        auto& w = m.reserve(3, sizeof(double) * rows);
        auto& s = m.reserve(4, sizeof(double) * columns);
        auto& a = m.reserve(5, sizeof(double) * size);
        auto& wa = m.reserve(6, sizeof(double) * size);
        auto& result = m.reserve(7, sizeof(double) * columns * columns);
        if (m.status) m.status = w.upload(weights, sizeof(double) * rows);
        if (m.status && scale) m.status = s.upload(scale, sizeof(double) * columns);
        auto bp = m.buffers.at(0)->address(), wp = w.address(), sp = scale ? s.address() : 0ULL;
        auto ap = a.address(), wap = wa.address();
        int symmetric = std::all_of(weights, weights + rows, [](double v) { return v >= 0; });
        void* args[] = {&bp, &wp, &sp, &ap, &wap, &rows, &columns, &symmetric};
        Launch shape; shape.grid_x = static_cast<unsigned>((size + 255) / 256);
        if (m.status) m.status = m.context.launch(kernel_source("dense_ops"), "tulpa_dense_weight", shape, args);
        if (symmetric) {
            using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, int, const double*,
                const double*, int, const double*, double*, int);
            auto fn = m.symbol<Fn>(Capability::Blas, "cublasDsyrk_v2");
            auto handle = m.context.handle(Capability::Blas, m.status);
            const double one = 1, zero = 0;
            if (m.status) m.status = Status::read("cuBLAS", fn(handle, 0, 1, columns, rows,
                &one, reinterpret_cast<double*>(ap), rows, &zero,
                reinterpret_cast<double*>(result.address()), columns), "cublasDsyrk_v2");
        } else if (m.status) m.status = gemm(m, true, false, columns, columns, rows,
            ap, rows, wap, rows, result.address());
        if (m.status && out) m.status = result.download(out, sizeof(double) * columns * columns);
        if (symmetric && m.status && out) for (int j = 0; j < columns; ++j) for (int i = 0; i < j; ++i)
            out[i + static_cast<std::size_t>(j) * columns] = out[j + static_cast<std::size_t>(i) * columns];
        return m.status;
    }
};

// ---- dense factor -----------------------------------------------------------

class CudaDenseFactor final : public DenseFactorRoute {
    Memory memory;
    FactorWorkspace own;
    std::vector<double> lower_cache, times_cache, prior_cache;
    Address factor_address = 0;
    int n = 0;
    bool laplace = false;
    std::string operation;
    void* solver = nullptr;
    void* stream = nullptr;
    double determinant = std::numeric_limits<double>::quiet_NaN();

    void begin(int order, const std::string& op, bool laplace_route) {
        memory.status = Status{};
        determinant = std::numeric_limits<double>::quiet_NaN();
        lower_cache.clear();
        factor_address = 0;
        n = order; operation = op; laplace = laplace_route;
    }
    bool factor_device() {
        if (laplace) {
            if (!stream && memory.status) memory.status = memory.context.create_stream(stream);
            if (!solver && memory.status) memory.status = memory.context.create_solver(solver, stream);
            bool positive = false;
            if (memory.status) memory.status = laplace_factor(factor_address, n, positive, solver, stream, own, operation);
            if (!memory.status || !positive) return false;
            lower_cache.assign(static_cast<std::size_t>(n) * n, 0.0);
            memory.status = memory.context.download(lower_cache.data(), factor_address, sizeof(double)*n*n);
            if (!memory.status) return false;
            for (int j = 0; j < n; ++j) for (int i = 0; i < j; ++i) lower_cache[i + static_cast<std::size_t>(j) * n] = 0;
            determinant = 0;
            for (int j = 0; j < n; ++j) {
                const double pivot = lower_cache[j + static_cast<std::size_t>(j) * n];
                if (!(pivot > 0) || !std::isfinite(pivot)) return false;
                determinant += 2*std::log(pivot);
            }
            return std::isfinite(determinant);
        }
        using SizeFn = int (TULPA_DENSE_CALL*)(void*, int, int, double*, int, int*);
        using FactorFn = int (TULPA_DENSE_CALL*)(void*, int, int, double*, int, double*, int, int*);
        auto size = memory.symbol<SizeFn>(Capability::Solver, "cusolverDnDpotrf_bufferSize");
        auto factor = memory.symbol<FactorFn>(Capability::Solver, "cusolverDnDpotrf");
        void* handle = memory.context.handle(Capability::Solver, memory.status);
        if (!factor_address) factor_address = memory.reserve(0, sizeof(double) * static_cast<std::size_t>(n) * n).address();
        auto& info = memory.reserve(1, sizeof(int));
        int workspace = 0, code = 0;
        if (memory.status) memory.status = Status::read("cuSOLVER", size(handle, 0, n,
            reinterpret_cast<double*>(factor_address), n, &workspace), "cusolverDnDpotrf_bufferSize");
        auto& work = memory.reserve(2, sizeof(double) * std::max(1, workspace));
        if (memory.status) memory.status = Status::read("cuSOLVER", factor(handle, 0, n,
            reinterpret_cast<double*>(factor_address), n, reinterpret_cast<double*>(work.address()),
            workspace, reinterpret_cast<int*>(info.address())), "cusolverDnDpotrf");
        if (memory.status) memory.status = info.download(&code, sizeof(code));
        if (code < 0) memory.status = Status::read("cuSOLVER", code, "potrf info");
        if (!memory.status || code > 0) return false;
        auto& diagonal = memory.reserve(3, sizeof(double) * n);
        auto a = factor_address, d = diagonal.address();
        void* args[] = {&a, &d, &n};
        Launch shape; shape.grid_x = (n + 255) / 256;
        memory.status = memory.context.launch(kernel_source("dense_ops"), "tulpa_dense_diagonal", shape, args);
        std::vector<double> host_diagonal(n);
        if (memory.status) memory.status = diagonal.download(host_diagonal.data(), sizeof(double) * n);
        determinant = 0;
        for (double v : host_diagonal) {
            if (!(v > 0) || !std::isfinite(v)) return false;
            determinant += 2 * std::log(v);
        }
        return static_cast<bool>(memory.status) && std::isfinite(determinant);
    }
public:
    ~CudaDenseFactor() override {
        if (!solver && !stream) return;
        auto& context = memory.context;
        Status status = context.push();
        if (status) {
            status.absorb(context.synchronize_stream(stream));
            if (solver) status.absorb(context.destroy_solver(solver));
            if (stream) status.absorb(context.destroy_stream(stream));
            context.pop();
        }
    }
    Status factorize(const double* matrix, int order, const std::string& op, bool laplace_route,
                     bool& positive) override {
        begin(order, op, laplace_route);
        auto& m = memory;
        positive = false;
        auto factor = [&] {
            auto& input = m.reserve(0, sizeof(double) * static_cast<std::size_t>(n) * n);
            factor_address = input.address();
            if (m.status) m.status = input.upload(matrix, sizeof(double) * static_cast<std::size_t>(n) * n);
            if (m.status) positive = factor_device();
        };
        if (laplace) {
            const auto pushed = m.context.push();
            m.status.absorb(pushed);
            if (m.status) factor();
            if (pushed) m.status.absorb(m.context.pop());
        } else {
            Lease lease(m.context, m.status);
            factor();
        }
        return m.status;
    }
    Status covariance(const double* times, int order, double sigma2, double phi, int kind,
                      double period, double nugget, bool& positive) override {
        begin(order, "temporal_gp", false);
        auto& m = memory;
        positive = false;
        {
            Lease lease(m.context, m.status);
            auto& input = m.reserve(4, sizeof(double) * n);
            const std::vector<double> requested(times, times + n);
            if (m.status && times_cache != requested) {
                m.status = input.upload(times, sizeof(double) * n);
                if (m.status) times_cache = requested;
            }
            auto& output = m.reserve(0, sizeof(double) * static_cast<std::size_t>(n) * n);
            auto t = input.address(), a = output.address();
            int count = n;
            double exp_lower = tulpa::math::EXP_ARG_MIN, exp_upper = tulpa::math::EXP_ARG_MAX;
            void* args[] = {&t, &a, &count, &sigma2, &phi, &kind, &period, &nugget, &exp_lower, &exp_upper};
            Launch shape; shape.grid_x = static_cast<unsigned>((static_cast<std::size_t>(n) * n + 255) / 256);
            if (m.status) m.status = m.context.launch(kernel_source("dense_covariance"),
                "tulpa_dense_covariance_kernel", shape, args);
            if (m.status) positive = factor_device();
        }
        return m.status;
    }
    Status factor_projected(DenseBasisRoute& gram, const double* prior, int order,
                            double prior_scale, double constant_augmentation, bool& positive) override {
        auto* basis = dynamic_cast<CudaDenseBasis*>(&gram);
        positive = false;
        if (!basis) return Status::read("CUDA", -1, "projected factor", "the gram product belongs to another backend");
        begin(order, operation, false);
        auto& m = memory;
        {
            Lease lease(m.context, m.status);
            const std::size_t count = static_cast<std::size_t>(n) * n;
            auto& prior_buffer = m.reserve(6, sizeof(double) * count);
            const std::vector<double> requested(prior, prior + count);
            if (prior_cache != requested) {
                if (m.status) m.status = prior_buffer.upload(prior, sizeof(double) * count);
                if (m.status) prior_cache = requested;
            }
            factor_address = basis->product();
            auto a = factor_address, b = prior_buffer.address();
            void* args[] = {&a, &b, &n, &prior_scale, &constant_augmentation};
            Launch shape; shape.grid_x = static_cast<unsigned>((count + 255) / 256);
            if (m.status) m.status = m.context.launch(kernel_source("dense_ops"), "tulpa_dense_add_prior", shape, args);
            if (m.status) positive = factor_device();
        }
        return m.status;
    }
    Status solve(const double* rhs, double* out, int columns, DenseSystem system) override {
        auto& m = memory;
        {
            Lease lease(m.context, m.status);
            auto& b = m.reserve(5, sizeof(double) * n * columns);
            if (m.status) m.status = b.upload(rhs, sizeof(double) * n * columns);
            const auto a = factor_address;
            if (laplace && system == DenseSystem::Symmetric) {
                auto factor = a, rhs_address = b.address();
                Launch shape; shape.grid_x = columns; shape.block_x = 128; shape.stream = stream;
                void* args[] = {&factor, &rhs_address, &n};
                if (m.status) m.status = m.context.launch(kernel_source("laplace_dense"), "laplace_dense_solve", shape, args);
                if (m.status) m.status = m.context.synchronize_stream(stream);
            } else if (system != DenseSystem::Symmetric) {
                if (m.status) m.status = triangular_solve(m, false,
                    system == DenseSystem::LowerTranspose, n, columns, a, n, b.address());
            } else {
                using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, const double*, int, double*, int, int*);
                auto fn = m.symbol<Fn>(Capability::Solver, "cusolverDnDpotrs");
                auto handle = m.context.handle(Capability::Solver, m.status);
                auto& info = *m.buffers.at(1);
                if (m.status) m.status = Status::read("cuSOLVER", fn(handle, 0, n, columns,
                    reinterpret_cast<double*>(a), n, reinterpret_cast<double*>(b.address()), n,
                    reinterpret_cast<int*>(info.address())), "cusolverDnDpotrs");
                int code = 0;
                if (m.status) m.status = info.download(&code, sizeof(code));
                if (code) m.status = Status::read("cuSOLVER", code, "potrs info");
            }
            if (m.status) m.status = b.download(out, sizeof(double) * n * columns);
        }
        return m.status;
    }
    Status lower(double* out) override {
        auto& m = memory;
        if (lower_cache.empty()) {
            lower_cache.assign(static_cast<std::size_t>(n) * n, 0.0);
            {
                Lease lease(m.context, m.status);
                if (m.status) m.status = m.context.download(lower_cache.data(), factor_address, sizeof(double) * n * n);
            }
            if (!m.status) { lower_cache.clear(); return m.status; }
            for (int j = 0; j < n; ++j) for (int i = 0; i < j; ++i) lower_cache[i + static_cast<std::size_t>(j) * n] = 0;
        }
        std::copy(lower_cache.begin(), lower_cache.end(), out);
        return m.status;
    }
    double log_determinant() const override { return determinant; }
};
}

std::unique_ptr<DenseBasisRoute> make_dense_basis() { return std::unique_ptr<DenseBasisRoute>(new CudaDenseBasis); }
std::unique_ptr<DenseFactorRoute> make_dense_factor() { return std::unique_ptr<DenseFactorRoute>(new CudaDenseFactor); }

Status cross_product(const double* left, int rows, int columns, const double* right,
                     int right_columns, double* out) {
    auto& m = thread_memory().cross;
    {
        Lease lease(m.context, m.status);
        const std::size_t a_bytes = sizeof(double) * static_cast<std::size_t>(rows) * columns;
        const std::size_t b_bytes = sizeof(double) * static_cast<std::size_t>(rows) * right_columns;
        const std::size_t c_bytes = sizeof(double) * static_cast<std::size_t>(columns) * right_columns;
        auto& a = m.reserve(0, a_bytes); auto& b = m.reserve(1, b_bytes); auto& c = m.reserve(2, c_bytes);
        if (m.status) m.status = a.upload(left, a_bytes);
        if (m.status) m.status = b.upload(right, b_bytes);
        if (m.status) m.status = gemm(m, true, false, columns, right_columns, rows,
            a.address(), rows, b.address(), rows, c.address());
        if (m.status) m.status = c.download(out, c_bytes);
    }
    return m.status;
}

Status cholesky_adjoint(const double* lower, const double* adjoint, double* out, int n) {
    auto& m = thread_memory().adjoint;
    {
        Lease lease(m.context, m.status);
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n) * n;
        auto& l = m.reserve(0, bytes);
        auto& g = m.reserve(1, bytes);
        auto& product = m.reserve(2, bytes);
        auto& symmetric = m.reserve(3, bytes);
        if (m.status) m.status = l.upload(lower, bytes);
        if (m.status) m.status = g.upload(adjoint, bytes);
        if (m.status) m.status = gemm(m, true, false, n, n, n,
            l.address(), n, g.address(), n, product.address());
        auto p = product.address(), s = symmetric.address();
        void* args[] = {&p, &s, &n};
        Launch shape; shape.grid_x = static_cast<unsigned>((static_cast<std::size_t>(n) * n + 255) / 256);
        if (m.status) m.status = m.context.launch(kernel_source("dense_ops"), "tulpa_dense_chol_project", shape, args);
        if (m.status) m.status = triangular_solve(m, false, true, n, n, l.address(), n, s);
        if (m.status) m.status = triangular_solve(m, true, false, n, n, l.address(), n, s);
        if (m.status) m.status = symmetric.download(out, bytes);
    }
    return m.status;
}

Status cholesky_batch(double* matrices, int n, int count) {
    Status s;
    const std::size_t stride = static_cast<std::size_t>(n) * n;
    auto& context = Context::instance();
    Lease lease(context, s);
    Buffer data(context, count * stride * sizeof(double), s);
    Buffer pointers(context, count * sizeof(Address), s);
    Buffer codes(context, count * sizeof(int), s);
    std::vector<Address> addresses(count);
    for (int b = 0; b < count; ++b) addresses[b] = data.address() + b * stride * sizeof(double);
    if (s) s = data.upload(matrices, data.size());
    if (s) s = pointers.upload(addresses.data(), pointers.size());
    if (s) s = context.factor(pointers.address(), n, count, codes.address());
    if (s) s = context.synchronize();
    std::vector<int> info(count);
    if (s) s = codes.download(info.data(), codes.size());
    if (s) for (int b = 0; b < count; ++b) {
        if (info[b]) { s = Status::read("cuSOLVER", info[b], "potrf matrix", "batch index " + std::to_string(b)); break; }
    }
    if (s) s = data.download(matrices, data.size());
    return s;
}

Status axpy(const double* x, double* y, double a, int n) {
    Status s;
    auto& context = Context::instance();
    Lease lease(context, s);
    Buffer dx(context, n * sizeof(double), s), dy(context, n * sizeof(double), s);
    if (s) s = dx.upload(x, dx.size());
    if (s) s = dy.upload(y, dy.size());
    Address px = dx.address(), py = dy.address();
    void* arguments[] = {&px, &py, &a, &n};
    Launch shape;
    shape.grid_x = (static_cast<unsigned>(n) + shape.block_x - 1) / shape.block_x;
    if (s) s = context.launch(kernel_source("axpy"), "tulpa_axpy", shape, arguments);
    if (s) s = context.synchronize();
    if (s) s = dy.download(y, dy.size());
    return s;
}

}
#undef TULPA_DENSE_CALL
