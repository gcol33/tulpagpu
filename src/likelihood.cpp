#include "routes.h"
#include <algorithm>
#include <numeric>

namespace tulpagpu {
namespace {

class CudaLikelihood final : public LikelihoodRoute {
    ObservationPlan plan;
    Context& context = Context::instance();
    Status status;
    enum Slot { X, Offset, Rowptr, Column, Weight, Colptr, Row, Transpose,
        Response, Constants, BetaStart, BetaCount, Parameters, Scores, Curvature,
        Partial, Gradient, Eta, Count };
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<void*> streams;
    int capacity = 0, blocks = 0;
    bool have_weights = false;

    unsigned long long address(Slot slot) const { return buffers[slot] ? buffers[slot]->address() : 0; }
    void allocate(Slot slot, std::size_t bytes) {
        buffers[slot].reset();
        buffers[slot] = std::make_unique<Buffer>(context,bytes,status);
    }
    template<class T> void upload(Slot slot, const std::vector<T>& data) {
        allocate(slot,data.size()*sizeof(T));
        if (status) status.absorb(buffers[slot]->upload(data.data(),data.size()*sizeof(T)));
    }
    void upload_projection() {
        upload(Rowptr,plan.rowptr); upload(Column,plan.column); upload(Weight,plan.weight);
        std::vector<int> colptr(plan.width+1,0), rows(plan.column.size());
        std::vector<double> transpose(plan.weight.size());
        for (int col : plan.column) ++colptr[col+1];
        std::partial_sum(colptr.begin(),colptr.end(),colptr.begin());
        auto next=colptr;
        for (int row=0; row<plan.n*plan.np; ++row)
            for (int j=plan.rowptr[row]; j<plan.rowptr[row+1]; ++j) {
                const int dest=next[plan.column[j]]++;
                rows[dest]=row; transpose[dest]=plan.weight[j];
            }
        upload(Colptr,colptr); upload(Row,rows); upload(Transpose,transpose);
    }
    void initialize(const tulpa::DeviceLikelihood& kernel, int chains, bool weights) {
        if (!buffers[X]) {
            upload(X,plan.x); upload(Offset,plan.offset); upload(Response,kernel.response);
            upload(BetaStart,plan.beta_start); upload(BetaCount,plan.beta_count);
            upload_projection();
            blocks = std::min(4096,(plan.n+255)/256);
        }
        if (chains>capacity || (weights && !have_weights)) {
            capacity = std::max(chains,capacity); have_weights = weights || have_weights;
            allocate(Parameters,static_cast<std::size_t>(capacity)*plan.width*sizeof(double));
            allocate(Scores,static_cast<std::size_t>(capacity)*plan.n*plan.np*sizeof(double));
            allocate(Partial,static_cast<std::size_t>(capacity)*blocks*sizeof(double));
            allocate(Gradient,static_cast<std::size_t>(capacity)*plan.width*sizeof(double));
            if (have_weights) {
                allocate(Curvature,static_cast<std::size_t>(capacity)*plan.n*plan.np*plan.np*sizeof(double));
                allocate(Eta,static_cast<std::size_t>(capacity)*plan.n*plan.np*sizeof(double));
            }
        }
        if (!buffers[Constants] || buffers[Constants]->size()!=kernel.constants.size()*sizeof(double))
            allocate(Constants,kernel.constants.size()*sizeof(double));
        while (status && static_cast<int>(streams.size())<chains) {
            void* stream = nullptr;
            status.absorb(context.create_stream(stream));
            if (stream) streams.push_back(stream);
        }
    }
public:
    explicit CudaLikelihood(const ObservationPlan& p) : plan(p), buffers(Count) {}
    ~CudaLikelihood() override {
        if (buffers[X]) {
            Status cleanup;
            Lease lease(context,cleanup);
            cleanup.absorb(context.synchronize());
            for (void* stream : streams) cleanup.absorb(context.destroy_stream(stream));
            buffers.clear();
        }
    }
    Status refresh_projection(const ObservationPlan& next) override {
        Status s;
        if (buffers[X]) {
            Lease lease(context,s);
            status={};
            if (next.x!=plan.x) { plan.x=next.x; upload(X,plan.x); }
            if (next.offset!=plan.offset) { plan.offset=next.offset; upload(Offset,plan.offset); }
            const bool changed=next.rowptr!=plan.rowptr || next.column!=plan.column || next.weight!=plan.weight;
            plan=next;
            if (changed) upload_projection();
            s.absorb(status);
        } else plan=next;
        return s;
    }
    Status evaluate(const tulpa::DeviceLikelihood& kernel, const std::vector<double>& parameters,
                    int chains, bool batched, bool weights, LikelihoodResult& result) override {
        Status s = context.supports(Capability::Compiler);
        if (s) s = context.supports(Capability::Blas);
        if (!s) return s;
        {
            Lease lease(context,s);
            status = {};
            initialize(kernel,chains,weights); s.absorb(status);
            if (s) s = buffers[Parameters]->upload(parameters.data(),parameters.size()*sizeof(double));
            if (s) s = buffers[Constants]->upload(kernel.constants.data(),kernel.constants.size()*sizeof(double));
            const std::string source = kernel.source+kernel_source("likelihood");
            auto launch = [&](int first, int count, void* stream) {
                if (!s) return;
                int n=plan.n, np=plan.np, width=plan.width;
                auto x=address(X), offset=address(Offset), rowptr=address(Rowptr);
                auto column=address(Column), projection=address(Weight), response=address(Response);
                auto constants=address(Constants), start=address(BetaStart), columns=address(BetaCount);
                auto par=address(Parameters)+static_cast<std::size_t>(first)*width*sizeof(double);
                auto score=address(Scores)+static_cast<std::size_t>(first)*n*np*sizeof(double);
                auto curvature=weights ? address(Curvature)+static_cast<std::size_t>(first)*n*np*np*sizeof(double) : 0;
                auto partial=address(Partial)+static_cast<std::size_t>(first)*blocks*sizeof(double);
                auto eta=weights ? address(Eta)+static_cast<std::size_t>(first)*n*np*sizeof(double) : 0;
                void* args[]={&n,&np,&width,&start,&columns,&x,&offset,&rowptr,&column,&projection,&response,&constants,&par,&score,&curvature,&partial,&eta};
                Launch shape; shape.grid_x=blocks; shape.grid_y=count; shape.stream=stream;
                s = context.launch(source,"tulpa_likelihood",shape,args);
                auto cp=address(Colptr), row=address(Row), wt=address(Transpose);
                auto gradient=address(Gradient)+static_cast<std::size_t>(first)*width*sizeof(double);
                void* scatter[]={&n,&np,&width,&cp,&row,&wt,&score,&gradient};
                shape.grid_x=std::min(width,4096);
                if (s) s = context.launch(source,"tulpa_likelihood_scatter",shape,scatter);
                for (int chain=first; s && chain<first+count; ++chain) {
                    std::size_t xoff=0;
                    for (int k=0; s && k<np; ++k) {
                        const int p=plan.beta_count[k];
                        if (p) s = context.gemv(address(X)+xoff*sizeof(double),
                            address(Scores)+(static_cast<std::size_t>(chain)*np+k)*n*sizeof(double),
                            address(Gradient)+(static_cast<std::size_t>(chain)*width+plan.beta_start[k])*sizeof(double),
                            n,p,true,1,stream);
                        xoff += static_cast<std::size_t>(n)*p;
                    }
                }
            };
            if (batched) launch(0,chains,streams[0]);
            else for (int chain=0; s && chain<chains; ++chain) launch(chain,1,streams[chain]);
            for (int chain=0; chain<(batched ? 1 : chains) && chain<static_cast<int>(streams.size()); ++chain)
                s.absorb(context.synchronize_stream(streams[chain]));
            std::vector<double> partial(static_cast<std::size_t>(chains)*blocks);
            result.gradient.resize(parameters.size()); result.value.assign(chains,0);
            if (s) s = buffers[Partial]->download(partial.data(),partial.size()*sizeof(double));
            if (s) s = buffers[Gradient]->download(result.gradient.data(),result.gradient.size()*sizeof(double));
            if (s) for (int chain=0; chain<chains; ++chain)
                result.value[chain]=std::accumulate(partial.begin()+chain*blocks,partial.begin()+(chain+1)*blocks,0.0);
            result.score.clear(); result.curvature.clear(); result.eta.clear();
            if (s && weights) {
                result.score.resize(static_cast<std::size_t>(chains)*plan.n*plan.np);
                result.curvature.resize(result.score.size()*plan.np);
                result.eta.resize(result.score.size());
                s = buffers[Scores]->download(result.score.data(),result.score.size()*sizeof(double));
                if (s) s = buffers[Curvature]->download(result.curvature.data(),result.curvature.size()*sizeof(double));
                if (s) s = buffers[Eta]->download(result.eta.data(),result.eta.size()*sizeof(double));
            }
        }
        s.absorb(status);
        return s;
    }
};
}

std::unique_ptr<LikelihoodRoute> make_likelihood(const ObservationPlan& plan) {
    return std::unique_ptr<LikelihoodRoute>(new CudaLikelihood(plan));
}

}
