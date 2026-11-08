// Tensor G2 persistent worker pool + best-effort affinity experiments.
//
// The benchmark deliberately separates per-inference worker creation from
// persistent workers. The pool is built BEFORE timing starts. Main thread
// participates in computation and 0..3 worker threads are notified by a CV.
//
// Linux cpuset restrictions are respected; pinning never expands available
// process affinity and the original main-thread CPU mask is restored.
#include "cpu_pool.h"
#include "kernel_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace g2 {
namespace {

using Clock = std::chrono::steady_clock;

struct CpuInfo {
    int id = -1;
    int score = 0;
};

int read_number(const std::string& path) {
    std::ifstream in(path);
    int value = 0;
    if (in >> value) return value;
    return 0;
}

std::vector<CpuInfo> allowed_cpus() {
    std::vector<CpuInfo> result;
#if defined(__linux__)
    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) == 0) {
        for (int i=0; i<CPU_SETSIZE; ++i) {
            if (!CPU_ISSET(i, &available)) continue;
            const std::string base =
                "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/";
            int score=read_number(base+"cpu_capacity");
            if (!score) score=read_number(base+"cpufreq/cpuinfo_max_freq");
            if (!score) score=read_number(base+"cpufreq/scaling_max_freq");
            result.push_back({i,score});
        }
    }
#endif
    std::stable_sort(result.begin(),result.end(),[](const CpuInfo& a,const CpuInfo& b) {
        if (a.score!=b.score) return a.score>b.score;
        return a.id>b.id;
    });
    return result;
}

bool pin_current_cpu(int id) {
#if defined(__linux__)
    if (id<0 || id>=CPU_SETSIZE) return false;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(id, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask)==0;
#else
    (void)id;
    return false;
#endif
}

struct RestoreAffinity {
#if defined(__linux__)
    cpu_set_t original{};
#endif
    bool captured=false;
    bool applied=false;
    explicit RestoreAffinity(int target) {
#if defined(__linux__)
        captured=sched_getaffinity(0,sizeof(original),&original)==0;
        if (captured) applied=pin_current_cpu(target);
#else
        (void)target;
#endif
    }
    ~RestoreAffinity() {
#if defined(__linux__)
        if (captured && applied)
            sched_setaffinity(0,sizeof(original),&original);
#endif
    }
};

class PersistentPool {
public:
    explicit PersistentPool(bool pin,
                            const std::vector<CpuInfo>& cpus)
      : pinned_(pin), cpus_(cpus) {
        for (int slot=0;slot<3;++slot) {
            threads_.emplace_back([this,slot] {worker(slot);});
        }
    }

    ~PersistentPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_=true;
        }
        start_cv_.notify_all();
        for (auto& thread:threads_) thread.join();
    }

    int successful_worker_pins() const {return pins_.load();}

    void run(const Matrix& matrix,const QuantizedInput& x,float* out,int count) {
        count=std::max(1,std::min(4,count));
        if(count==1) {
            dotprod_gemv_rows(matrix,x,out,0,matrix.rows);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            matrix_=&matrix;
            activations_=&x;
            output_=out;
            running_=count;
            completed_=0;
            ++generation_;
        }
        start_cv_.notify_all();
        dotprod_gemv_rows(matrix,x,out,0,matrix.rows/count);
        std::unique_lock<std::mutex> lock(mutex_);
        end_cv_.wait(lock,[this] {return completed_ == running_-1;});
    }

private:
    bool pinned_=false;
    std::vector<CpuInfo> cpus_;
    std::vector<std::thread> threads_;
    std::atomic<int> pins_{0};
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable end_cv_;
    bool stop_=false;
    uint64_t generation_=0;
    int running_=1,completed_=0;
    const Matrix* matrix_=nullptr;
    const QuantizedInput* activations_=nullptr;
    float* output_=nullptr;

    void worker(int slot) {
        if (pinned_ && !cpus_.empty()) {
            const int cpu=cpus_[size_t(slot+1)%cpus_.size()].id;
            if (pin_current_cpu(cpu)) ++pins_;
        }
        uint64_t last_seen=0;
        std::unique_lock<std::mutex> lock(mutex_);
        while(true) {
            start_cv_.wait(lock,[&] {return stop_ || generation_!=last_seen;});
            if(stop_) return;
            last_seen=generation_;
            if (slot>=running_-1) continue;
            const Matrix* m=matrix_;
            const QuantizedInput* x=activations_;
            float* dst=output_;
            const int count=running_;
            const int first=int(int64_t(m->rows)*(slot+1)/count);
            const int last=int(int64_t(m->rows)*(slot+2)/count);
            lock.unlock();
            dotprod_gemv_rows(*m,*x,dst,first,last);
            lock.lock();
            ++completed_;
            end_cv_.notify_one();
        }
    }
};

template<class F>
double median_ms(F&& call,int repetitions) {
    // Avoid mixing cold-start/pool creation and steady-state latency.
    call();
    std::vector<double> measured;
    const int batches=3;
    for (int j=0;j<batches;++j) {
        const auto start=Clock::now();
        for (int i=0;i<repetitions;++i) call();
        const auto end=Clock::now();
        measured.push_back(
            std::chrono::duration<double,std::milli>(end-start).count()/repetitions);
    }
    std::sort(measured.begin(),measured.end());
    return measured[measured.size()/2];
}

} // namespace

std::string cpu_pool_benchmark_json(
    const Matrix& matrix,const QuantizedInput& quantized,
    const float* reference,int repetitions,double activation_quant_ms) {
    const auto cpus=allowed_cpus();
    double reference_energy=0;
    for(int row=0;row<matrix.rows;++row)
        reference_energy+=double(reference[row])*double(reference[row]);
    std::vector<float> output(size_t(matrix.rows),0.0f);
    std::ostringstream out;
    out<<std::fixed<<std::setprecision(6);
    out<<"{\"available_allowed_cpus\":[";
    for(size_t i=0;i<cpus.size();++i) {
        if(i)out<<",";
        out<<"{\"cpu\":"<<cpus[i].id<<",\"capacity_or_max_freq\":"<<cpus[i].score<<"}";
    }
    out<<"],\"variants\":[";
    bool first=true;
    for(bool pin:{false,true}) {
        if(pin && cpus.empty())continue;
        PersistentPool pool(pin,cpus);
        for(int count:{1,2,4}) {
            // Restore calling task affinity after every variant.
            RestoreAffinity guard(pin && !cpus.empty()?cpus[0].id:-1);
            const bool main_pinned=pin && guard.applied;
            const double elapsed=median_ms(
                [&] {pool.run(matrix,quantized,output.data(),count);},repetitions);
            double err2=0.0;
            for(int row=0;row<matrix.rows;++row) {
                const double delta=double(output[row])-double(reference[row]);
                err2+=delta*delta;
            }
            const double nrmse=std::sqrt(err2/std::max(reference_energy,1e-30));
            if(!first)out<<",";
            first=false;
            out<<"{\"threads\":"<<count
               <<",\"strategy\":\""<<(pin?"persistent_affinity":"persistent")<<"\""
               <<",\"main_affinity_applied\":"<<(main_pinned?"true":"false")
               <<",\"worker_affinity_successes\":"<<pool.successful_worker_pins()
               <<",\"median_kernel_wall_ms\":"<<elapsed
               <<",\"total_with_activation_quant_ms\":"<<elapsed+activation_quant_ms
               <<",\"nrmse\":"<<nrmse
               <<",\"pass\":"<<(std::isfinite(nrmse)&&nrmse<0.05?"true":"false")
               <<"}";
        }
    }
    out<<"]}";
    return out.str();
}
}  // namespace g2
