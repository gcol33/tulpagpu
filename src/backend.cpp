#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include "routes.h"
#include <thread>

namespace tulpagpu {
namespace {

class CudaBackend final : public tulpa::device::Backend {
public:
    Status initialize() override { return Context::instance().initialize(); }
    Info info() override { return Context::instance().info(); }
    Status supports(Capability capability) override { return Context::instance().supports(capability); }
    std::vector<tulpa::device::Threshold> thresholds() override { return tulpagpu::thresholds(); }
    Status memory(std::size_t& free, std::size_t& total) override {
        auto& context = Context::instance();
        Status status = context.initialize();
        if (status) {
            Lease lease(context, status);
            if (status) status = context.memory_available(free, total);
        }
        return status;
    }
    std::unique_ptr<DenseBasisRoute> dense_basis() override { return make_dense_basis(); }
    std::unique_ptr<DenseFactorRoute> dense_factor() override { return make_dense_factor(); }
    std::unique_ptr<DenseScatterRoute> dense_scatter() override { return make_dense_scatter(); }
    std::unique_ptr<SparseFactorRoute> sparse_factor() override { return make_sparse_factor(); }
    std::unique_ptr<LikelihoodRoute> likelihood(const ObservationPlan& plan) override {
        return make_likelihood(plan);
    }
    Status cross_product(const double* left, int rows, int columns, const double* right,
                         int right_columns, double* out) override {
        return tulpagpu::cross_product(left, rows, columns, right, right_columns, out);
    }
    Status cholesky_adjoint(const double* lower, const double* adjoint, double* out, int n) override {
        return tulpagpu::cholesky_adjoint(lower, adjoint, out, n);
    }
    Status cholesky_batch(double* matrices, int n, int count) override {
        return tulpagpu::cholesky_batch(matrices, n, count);
    }
    Status axpy(const double* x, double* y, double a, int n) override {
        return tulpagpu::axpy(x, y, a, n);
    }
};

// Lives for the process: routes handed to the engine refer to this object's code.
CudaBackend& backend() { static CudaBackend* value = new CudaBackend; return *value; }

void check(const Status& status) { if (!status) Rcpp::stop(status.message); }

std::vector<tulpa::device::Threshold> read_thresholds(const Rcpp::DataFrame& table) {
    Rcpp::CharacterVector operation = table["operation"];
    Rcpp::IntegerVector variant = table["variant"];
    Rcpp::NumericVector size = table["threshold"];
    Rcpp::NumericVector batch_size = table.containsElementNamed("batch_threshold")
        ? Rcpp::NumericVector(table["batch_threshold"]) : Rcpp::NumericVector(table.nrows(), 0.0);
    std::vector<tulpa::device::Threshold> thresholds;
    for (R_xlen_t i = 0; i < operation.size(); ++i) {
        if (Rcpp::IntegerVector::is_na(variant[i]) || std::isnan(size[i]) || size[i] < 0 ||
            std::isnan(batch_size[i]) || batch_size[i] < 0)
            Rcpp::stop("invalid device threshold");
        thresholds.push_back({Rcpp::as<std::string>(operation[i]), variant[i], size[i], batch_size[i]});
    }
    return thresholds;
}
}
}

// [[Rcpp::export]]
void cpp_gpu_register(Rcpp::DataFrame table, Rcpp::List sources) {
    Rcpp::CharacterVector names = sources.names();
    std::map<std::string, std::string> kernels;
    for (R_xlen_t i = 0; i < sources.size(); ++i)
        kernels.emplace(Rcpp::as<std::string>(names[i]), Rcpp::as<std::string>(sources[i]));
    tulpagpu::configure(tulpagpu::read_thresholds(table), kernels);
    auto attach = reinterpret_cast<tulpa::device::RegisterBackendFn>(
        R_GetCCallable("tulpa", "tulpa_device_register"));
    const auto layout = tulpa::device::backend_layout();
    char reason[512] = {};
    auto* host = attach(&tulpagpu::backend(), &layout, reason, sizeof(reason));
    if (!host) Rcpp::stop(reason);
    tulpagpu::set_host(host);
}

// [[Rcpp::export]]
void cpp_gpu_unregister() {
    auto detach = reinterpret_cast<tulpa::device::UnregisterBackendFn>(
        R_GetCCallable("tulpa", "tulpa_device_unregister"));
    detach(&tulpagpu::backend());
}

// [[Rcpp::export]]
void cpp_gpu_thresholds(Rcpp::DataFrame table) {
    tulpagpu::configure_thresholds(tulpagpu::read_thresholds(table));
}

// [[Rcpp::export]]
Rcpp::List cpp_gpu_info() {
    auto info = tulpagpu::Context::instance().info();
    Rcpp::CharacterVector libraries = Rcpp::wrap(info.libraries);
    Rcpp::CharacterVector reasons = Rcpp::wrap(info.library_reasons);
    Rcpp::CharacterVector names = Rcpp::CharacterVector::create("driver", "cusolver", "cublas", "nvrtc", "cudss");
    libraries.names() = names; reasons.names() = names;
    return Rcpp::List::create(Rcpp::Named("available") = info.available,
        Rcpp::Named("compiled") = info.compiled,
        Rcpp::Named("backend") = info.available ? "cuda" : "cpu",
        Rcpp::Named("reason") = info.reason,
        Rcpp::Named("device_count") = info.count,
        Rcpp::Named("name") = info.name,
        Rcpp::Named("memory_mb") = info.memory_mb,
        Rcpp::Named("compute_capability") = info.compute_capability,
        Rcpp::Named("driver_version") = info.driver_version,
        Rcpp::Named("initializations") = info.init_count,
        Rcpp::Named("libraries") = libraries,
        Rcpp::Named("library_reasons") = reasons);
}

// [[Rcpp::export]]
Rcpp::List cpp_gpu_test(std::string action, Rcpp::NumericVector input,
                        std::string source = "", int threads = 8) {
    using tulpagpu::Status;
    auto& context = tulpagpu::Context::instance();
    Status status;
    std::vector<double> output(input.begin(), input.end());
    if (action == "status") tulpagpu::check(Status::read("CUDA", 1, "test_status", "invalid value"));
    if (action == "init") {
        if (threads < 1 || threads > 64) Rcpp::stop("threads must be in [1, 64]");
        std::vector<Status> statuses(threads);
        std::vector<std::thread> workers;
        for (int i = 0; i < threads; ++i) workers.emplace_back([&, i] { statuses[i] = context.initialize(); });
        for (auto& worker : workers) worker.join();
        for (const auto& s : statuses) status.absorb(s);
    } else if (action == "buffer" || action == "bounds" || action == "compile") {
        status = context.initialize();
        {
            tulpagpu::Lease lease(context, status);
            if (action == "compile") {
                void* module = nullptr; void* kernel = nullptr;
                if (status) status = context.compile(source, "tulpa_axpy", module, kernel);
                if (module) status.absorb(context.unload(module));
            } else {
                tulpagpu::Buffer buffer(context, output.size() * sizeof(double), status);
                if (status) status = buffer.upload(output.data(), buffer.size() + (action == "bounds" ? 1 : 0));
                if (status) status = buffer.download(output.data(), buffer.size());
            }
        }
    } else if (action != "init") Rcpp::stop("unknown device test action");
    tulpagpu::check(status);
    return Rcpp::List::create(Rcpp::Named("output") = output,
        Rcpp::Named("initializations") = context.info().init_count);
}
