#include "routes.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace tulpagpu {
namespace {

struct Request {
    unsigned long long design, vectors, output;
    int rows, columns;
    void* stream;
    Status status;
};

struct Workspace {
    Context& context = Context::instance();
    Status status;
    std::unique_ptr<Buffer> parameters;
    std::size_t capacity = 0;
    void run(const std::vector<Request*>& tasks) {
        if (tasks.empty()) return;
        Frame frame(context, status);
        const std::size_t bytes = tasks.size()*3*sizeof(unsigned long long);
        if (bytes > capacity) {
            parameters.reset(new Buffer(context, bytes, status)); capacity = bytes;
        }
        std::vector<unsigned long long> addresses;
        for (auto* t : tasks) {
            addresses.push_back(t->design); addresses.push_back(t->vectors); addresses.push_back(t->output);
        }
        if (status) status.absorb(parameters->upload(addresses.data(), bytes));
        auto p = parameters->address();
        int rows = tasks[0]->rows, columns = tasks[0]->columns;
        Launch shape; shape.block_x = 128;
        shape.grid_x = static_cast<unsigned>((static_cast<unsigned long long>(columns)*columns + 127)/128);
        shape.grid_y = static_cast<unsigned>(tasks.size());
        shape.stream = tasks.size() == 1 ? tasks[0]->stream : nullptr;
        void* args[] = {&p, &rows, &columns};
        { Lease lease(context, status);
          if (status) status.absorb(context.launch(kernel_source("laplace_scatter"), "laplace_scatter", shape, args)); }
        if (status) status.absorb(context.synchronize_stream(shape.stream));
        for (auto* t : tasks) t->status = status;
        auto policy = host().policy();
        policy->record("laplace_scatter_launch", true);
        if (tasks.size() > 1) policy->record("laplace_scatter_batched_launch", true);
    }
    ~Workspace() {
        if (!parameters) return;
        Frame frame(context, status);
        if (frame.pushed) parameters.reset();
    }
};

class CudaDenseScatter final : public DenseScatterRoute {
    Context& context = Context::instance();
    Status status;
    const double* design = nullptr;
    int rows = 0, columns = 0;
    bool uploaded = false;
    std::unique_ptr<Buffer> data, vectors, output;
    void* stream = nullptr;
    Workspace workspace;
    std::vector<double> packed, result;
public:
    ~CudaDenseScatter() override {
        if (!stream && !data && !vectors && !output) return;
        Frame frame(context, status);
        if (frame.pushed) {
            if (stream) status.absorb(context.synchronize_stream(stream));
            data.reset(); vectors.reset(); output.reset();
            if (stream) status.absorb(context.destroy_stream(stream));
        }
    }
    void bind(const double* row_major_design, int r, int c) override {
        if ((rows != r || columns != c) && (data || vectors || output)) {
            Frame frame(context, status);
            if (frame.pushed) { data.reset(); vectors.reset(); output.reset(); }
        }
        design = row_major_design; rows = r; columns = c;
        uploaded = false;
    }
    Status apply(const double* scores, const double* weights, double* gradient,
                 double* lower, BatchView* batch) override {
        Frame frame(context, status);
        if (status && !stream) status.absorb(context.create_stream(stream));
        const std::size_t count = static_cast<std::size_t>(columns)*(columns + 1ULL);
        if (!data) {
            data.reset(new Buffer(context, static_cast<std::size_t>(rows)*columns*sizeof(double), status));
            vectors.reset(new Buffer(context, 2ULL*rows*sizeof(double), status));
            output.reset(new Buffer(context, count*sizeof(double), status));
        }
        if (status && !uploaded) {
            status.absorb(data->upload(design, data->size())); uploaded = bool(status);
        }
        packed.resize(2ULL*rows);
        std::copy_n(scores, rows, packed.begin());
        std::copy_n(weights, rows, packed.begin()+rows);
        if (status) status.absorb(vectors->upload(packed.data(), vectors->size()));
        Request task{data->address(), vectors->address(), output->address(), rows, columns, stream};
        if (status && batch) {
            const std::string key = "dense_scatter:" + std::to_string(rows) + ":" + std::to_string(columns);
            auto group = std::static_pointer_cast<Workspace>(batch->claim(key,
                []() -> std::shared_ptr<void> { return std::make_shared<Workspace>(); },
                [](const std::shared_ptr<void>&) { return true; }));
            if (!group) status = Status::read("CUDA", -1, "dense scatter", "no shared scatter workspace");
            else status.absorb(batch->submit(group.get(), &task, [group](const std::vector<void*>& items) {
                std::vector<Request*> tasks;
                for (auto* t : items) tasks.push_back(static_cast<Request*>(t));
                group->run(tasks);
            }));
        } else if (status) workspace.run({&task});
        status.absorb(task.status);
        result.resize(count);
        if (status) status.absorb(output->download(result.data(), output->size()));
        if (status) for (double x : result) if (!std::isfinite(x)) {
            status = Status::read("CUDA", -1, "dense scatter", "nonfinite observation contribution"); break;
        }
        if (status) {
            std::copy_n(result.data(), static_cast<std::size_t>(columns)*columns, lower);
            std::copy_n(result.data()+static_cast<std::size_t>(columns)*columns, columns, gradient);
        }
        return status;
    }
};
}

std::unique_ptr<DenseScatterRoute> make_dense_scatter() {
    return std::unique_ptr<DenseScatterRoute>(new CudaDenseScatter);
}

}
