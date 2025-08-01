/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <array>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <charconv>

#include <cuda/atomic>

#include <madrona/mw_gpu.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/cuda_utils.hpp>
#include <madrona/tracing.hpp>
#include <madrona/json.hpp>
#include <madrona/render/asset_processor.hpp>

#include "cpp_compile.hpp"

#define KHRONOS_STATIC
#include <span>
// #define MADRONA_FAST_BVH

using bytes = std::span<const std::byte>;

template <>
struct std::hash<bytes>
{
    std::size_t operator()(const bytes& x) const noexcept
    {
        return std::hash<std::string_view>{}({reinterpret_cast<const char*>(x.data()), x.size()});
    }
};


// Wrap GPU headers in the mwGPU namespace. This is a weird situation where
// the CPU madrona headers are available but we need access to the GPU
// headers in order to do initial setup.
namespace mwGPU {
using namespace madrona;
#include "device/include/madrona/mw_gpu/const.hpp"
#include "device/include/madrona/memory.hpp"
#include "device/include/madrona/mw_gpu/host_print.hpp"
#include "device/include/madrona/mw_gpu/tracing.hpp"
#include "device/include/madrona/bvh.hpp"

namespace madrona::mwGPU {

class HostPrintCPU {
public:
    inline HostPrintCPU(CUdevice cu_gpu)
        : channel_([cu_gpu]() {
              CUdeviceptr channel_devptr;
              REQ_CU(CudaDynamicLoader::cuMemAllocManaged(&channel_devptr,
                                       sizeof(HostPrint::Channel),
                                       CU_MEM_ATTACH_GLOBAL));

              REQ_CU(CudaDynamicLoader::cuMemAdvise((CUdeviceptr)channel_devptr,
                                 sizeof(HostPrint::Channel),
                                 CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
              REQ_CU(CudaDynamicLoader::cuMemAdvise((CUdeviceptr)channel_devptr,
                                 sizeof(HostPrint::Channel),
                                 CU_MEM_ADVISE_SET_ACCESSED_BY, CU_DEVICE_CPU));

              REQ_CU(CudaDynamicLoader::cuMemAdvise(channel_devptr, sizeof(HostPrint::Channel),
                                 CU_MEM_ADVISE_SET_ACCESSED_BY, cu_gpu));

              auto ptr = (HostPrint::Channel *)channel_devptr;
              ptr->signal.store(0, cuda::std::memory_order_release);

              return ptr;
          }()),
          thread_([this]() {
              printThread();
          })
    {}

    HostPrintCPU(HostPrintCPU &&o) = delete;

    inline ~HostPrintCPU()
    {
        channel_->signal.store(-1, cuda::std::memory_order_release);
        thread_.join();
        REQ_CU(CudaDynamicLoader::cuMemFree((CUdeviceptr)channel_));
    }

    inline void * getChannelPtr()
    {
        return channel_;
    }

private:
    inline void printThread()
    {
        using namespace std::chrono_literals;
        using cuda::std::memory_order_acquire;
        using cuda::std::memory_order_relaxed;
        using FmtType = HostPrint::FmtType;

        const auto reset_duration = 1ms;
        const auto max_duration = 1s;

        auto cur_duration = reset_duration;

        while (true) {
            auto signal = channel_->signal.load(memory_order_acquire);
            if (signal == 0) {
                std::this_thread::sleep_for(cur_duration);

                cur_duration *= 2;
                if (cur_duration > max_duration) {
                    cur_duration = max_duration;
                }
                continue;
            } else if (signal == -1) {
                break;
            }


            std::cout << "GPU debug print:" << std::endl;
            std::string_view print_str = channel_->buffer;
            size_t buffer_offset = print_str.length() + 1;
            size_t str_offset = 0;

            CountT cur_arg = 0;
            while (str_offset < print_str.size()) {
                size_t pos = print_str.find("{}", str_offset);
                if (pos == print_str.npos) {
                    std::cout << print_str.substr(str_offset);
                    break;
                }

                std::cout << print_str.substr(str_offset, pos - str_offset);

                assert(cur_arg < channel_->numArgs);
                FmtType type = channel_->args[cur_arg];
                switch (type) {
                case FmtType::I32: {
                    int32_t v;
                    memcpy(&v, &channel_->buffer[buffer_offset],
                           sizeof(int32_t));
                    buffer_offset += sizeof(uint32_t);
                    std::cout << v;
                } break;
                case FmtType::U32: {
                    uint32_t v;
                    memcpy(&v, &channel_->buffer[buffer_offset],
                           sizeof(uint32_t));
                    buffer_offset += sizeof(uint32_t);
                    std::cout << v;
                } break;
                case FmtType::I64: {
                    int64_t v;
                    memcpy(&v, &channel_->buffer[buffer_offset],
                           sizeof(int64_t));
                    buffer_offset += sizeof(int64_t);
                    std::cout << v;
                } break;
                case FmtType::U64: {
                    uint64_t v;
                    memcpy(&v, &channel_->buffer[buffer_offset],
                           sizeof(uint64_t));
                    buffer_offset += sizeof(uint64_t);
                    std::cout << v;
                } break;
                case FmtType::Float: {
                    float v;
                    memcpy(&v, &channel_->buffer[buffer_offset],
                           sizeof(float));
                    buffer_offset += sizeof(float);
                    std::cout << v;
                } break;
                case FmtType::Ptr: {
                    void *v;
                    memcpy(&v, &channel_->buffer[buffer_offset],
                           sizeof(void *));
                    buffer_offset += sizeof(void *);
                    std::cout << v;
                } break;
                }
                
                cur_arg++;
                str_offset = pos + 2;
            }

            std::cout << std::flush;

            channel_->signal.store(0, memory_order_relaxed);
            cur_duration = reset_duration;
        }
    }

    HostPrint::Channel *channel_;
    std::thread thread_;
};

#ifdef MADRONA_TRACING
class DeviceTracingManager {
public:
    inline DeviceTracingManager(void *dev_ptr)
        : device_tracing_((DeviceTracing *)dev_ptr), steps_(0)
    {
        readback_ = (DeviceTracing *)
            ::madrona::cu::allocReadback(sizeof(DeviceTracing) * max_log_steps_);
    }

    // async memcpy on the critical path, low overhead but can still impact the overall running time
    inline void transferLogToCPU()
    {
        if (steps_ < max_log_steps_) {
            REQ_CUDA(cudaMemcpyAsync(readback_ + steps_, device_tracing_, sizeof(DeviceTracing),
                                        cudaMemcpyDeviceToHost));
            steps_ += 1;
        }
        // can also process data on the data paths to save memory
        // device_logs_cpu_.insert(device_logs_cpu_.end(), readback_->device_logs_, readback_->device_logs_ + readback_->getIndex());
    }

    inline ~DeviceTracingManager()
    {
        size_t num_logs = 0;
        for (size_t i = 0; i < steps_; i++) {
            auto log_index = (readback_ + i)->getIndex();
            num_logs += log_index > 0 ? log_index : 0;
        }
        device_logs_cpu_ = new DeviceTracing::DeviceLog[num_logs];
        
        num_logs = 0;
        for (size_t i = 0; i < steps_; i++) {
            auto log_index = (readback_ + i)->getIndex();
            if (log_index <= 0) continue;
            std::memcpy(device_logs_cpu_ + num_logs, (readback_ + i)->device_logs_, log_index * sizeof(DeviceTracing::DeviceLog));
            num_logs += log_index;
        }
        ::madrona::WriteToFile<DeviceTracing::DeviceLog>(
            device_logs_cpu_, num_logs,
            "/tmp/", "_madrona_device_tracing");

        ::madrona::cu::deallocCPU(readback_);
        delete[] device_logs_cpu_;
    }

private:
    DeviceTracing *device_tracing_;
    DeviceTracing *readback_;
    // at most first 100 steps will be recorded for saving memory
    const static size_t max_log_steps_ = 100;
    size_t steps_;
    DeviceTracing::DeviceLog* device_logs_cpu_;
};
#endif

}
}

#include "device/include/madrona/mw_gpu/megakernel_consts.hpp"

namespace madrona {

static void setCudaHeapSize()
{
    char *disable_heap_size = getenv("MADRONA_DISABLE_CUDA_HEAP_SIZE");

    if (disable_heap_size && disable_heap_size[0] == '1') {
        return;
    } else {
        // FIXME size limit for device side malloc:
        size_t heap_size = 4ul*1024ul*1024ul*1024ul;

        char* user_heap_size = getenv("MADRONA_MWGPU_DEVICE_HEAP_SIZE");
        if (user_heap_size) {
            heap_size = strtoul(user_heap_size,nullptr,10);
        }

        cudaError_t result = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);
        // FIXME: Ingore 'cudaErrorInvalidValue' errors, because it would occur systematically if performed after
        // launching any kernel that uses 'malloc' or 'free' system calls on this device. Anyway, it is not a big
        // deal and should not be considered as a failure. For reference, see:
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
        if (result == cudaErrorInvalidValue) {
            cudaGetLastError();
        }
        else
        {
            REQ_CUDA(result);
        }
    }
}

struct FreeQueue {
    struct Reserve {
        void *addr;
        uint64_t numBytes;
        uint64_t numReserveBytes;
    };

    std::vector<void *> toFree;
    std::vector<Reserve> reserveToFree;
};

using HostChannel = mwGPU::madrona::mwGPU::HostChannel;
using HostAllocInit = mwGPU::madrona::mwGPU::HostAllocInit;
using HostPrint = mwGPU::madrona::mwGPU::HostPrint;
using HostPrintCPU = mwGPU::madrona::mwGPU::HostPrintCPU;
using DeviceTracingManager = mwGPU::madrona::mwGPU::DeviceTracingManager;

namespace consts {
static constexpr uint32_t numEntryQueueThreads = 512;
}

using GPUImplConsts = mwGPU::madrona::mwGPU::GPUImplConsts;

enum class ExecutorMode {
    JobSystem,
    TaskGraph,
};

struct GPUCompileResults {
    CUmodule mod;
    std::string initECSName;
    std::string initWorldsName;
    std::string initTasksName;
};

struct MegakernelConfig {
    uint32_t numThreads;
    uint32_t numBlocksPerSM;
    uint32_t numSMs;
};

struct TimingData {
    float totalTime;
    float buildTimeRatio;
    float traceTimeRatio;

    float numTLASTraces;
    float numBLASTraces;
    float tlasRatio;

    uint64_t memoryUsage;
};

struct BVHKernels {
    uint32_t numSMs;
    CUmodule mod;

    // Initializes BVH internal data
    CUfunction init;

    // Allocates internal nodes
    CUfunction allocInternalNodes;

    // Entry point for creating the initial tree
    CUfunction buildFast;

    CUfunction buildSlow;

    // Entry point for optimizing the initial tree
    CUfunction optFast;

    // Debugging function
    CUfunction debug;

    // Raycasts
    CUfunction raycast;

    // Walks up the tree and constructs AABBs from the LBVH
    CUfunction constructAABBs;

    CUfunction widenTree;

    CUevent allocEvent;
    CUevent buildEvent;
    CUevent widenEvent;
    CUevent traceEvent;
    CUevent stopEvent;

    std::vector<TimingData> recordedTimings;

    mwGPU::madrona::KernelTimingInfo *timingInfo;

    uint32_t raycastRGBD;

    render::MeshBVHData meshBVHData;
    render::MaterialData materialData;
};

struct GPUKernels {
    CUmodule mod;
    HeapArray<CUfunction> megakernels;
    CUfunction computeGPUImplConsts;
    CUfunction initECS;
    CUfunction initWorlds;
    CUfunction initTasks;
    CUfunction queueUserInit;
    CUfunction queueUserRun;
    CUfunction initBVHParams;
    CUfunction destroyECS;
};

struct BVHKernelCache {
    HeapArray<char> data;
    const void *cubinStart;
    size_t numCubinBytes;
};

struct MegakernelCache {
    HeapArray<char> data;
    const void *cubinStart;
    size_t numCubinBytes;
    const char *initECSName;
    const char *initWorldsName;
    const char *initTasksName;
};

struct TaskGraphsState {
    HeapArray<CUfunction> megakernels;
    HeapArray<MegakernelConfig> megakernelConfigs;
    uint32_t numTaskgraphs;
};

struct GPUEngineState {
    void *stateBuffer;

    std::thread allocatorThread;
    HostChannel *hostAllocatorChannel;
    std::unique_ptr<HostPrintCPU> hostPrint;
#ifdef MADRONA_TRACING
    std::unique_ptr<DeviceTracingManager> deviceTracing;
#endif

    HeapArray<void *> exportedColumns;

    FreeQueue *freeQueue;
};

struct MWCudaLaunchGraph::Impl {
    CUgraphExec runGraph;
    bool enableRaytracing;

    CUevent start;
    CUevent end;

    const char *statName;
    uint32_t timingGroupIndex;
};

struct TimingGroup {
    const char *statName;
    std::vector<float> recordedTimings;
};

struct MWCudaExecutor::Impl {
    cudaStream_t cuStream;
    CUmodule cuModule;
    GPUEngineState engineState; 
    TaskGraphsState taskGraphsState;
    bool enableRaycasting;
    BVHKernels bvhKernels;
    CUfunction destroyKernel;

    uint32_t numWorlds;
    uint32_t renderOutputWidth;
    uint32_t renderOutputHeight;
    uint32_t sharedMemPerSM;

    std::vector<TimingGroup> timingGroups;
};

static void getUserEntries(const char *entry_class, CUmodule mod,
                           const char **compile_flags,
                           uint32_t num_compile_flags,
                           CUfunction *init_out, CUfunction *run_out)
{
    static const char mangle_code_postfix[] = R"__(
#include <cstdint>

namespace madrona { namespace mwGPU {

template <typename T> __global__ void submitInit(uint32_t, void *) {}
template <typename T> __global__ void submitRun(uint32_t) {}

} }
)__";

    static const char init_template[] =
        "::madrona::mwGPU::submitInit<::";
    static const char run_template[] =
        "::madrona::mwGPU::submitRun<::";

    std::string_view entry_view(entry_class);

    // If user prefixed with ::, trim off as it will be added later
    if (entry_view[0] == ':' && entry_view[1] == ':') {
        entry_view = entry_view.substr(2);
    }

    std::string fwd_declare; 

    // Find all namespace separators
    int num_namespaces = 0;
    size_t prev_off = 0, off = 0;
    while ((off = entry_view.find("::", prev_off)) != std::string_view::npos) {
        auto ns_view = entry_view.substr(prev_off, off - prev_off);

        fwd_declare += "namespace ";
        fwd_declare += ns_view;
        fwd_declare += " { ";

        prev_off = off + 2;
        num_namespaces++;
    }

    auto class_view = entry_view.substr(prev_off);
    if (class_view.size() == 0) {
        FATAL("Invalid entry class name\n");
    }

    fwd_declare += "class ";
    fwd_declare += class_view;
    fwd_declare += "; ";

    for (int i = 0; i < num_namespaces; i++) {
        fwd_declare += "} ";
    }

    std::string mangle_code = std::move(fwd_declare);
    mangle_code += mangle_code_postfix;

    std::string init_name = init_template;
    init_name += entry_view;
    init_name += ">";

    std::string run_name = run_template;
    run_name += entry_view;
    run_name += ">";

    nvrtcProgram prog;
    REQ_NVRTC(CudaDynamicLoader::nvrtcCreateProgram(&prog, mangle_code.c_str(), "mangle.cpp",
                                 0, nullptr, nullptr));

    REQ_NVRTC(CudaDynamicLoader::nvrtcAddNameExpression(prog, init_name.c_str()));
    REQ_NVRTC(CudaDynamicLoader::nvrtcAddNameExpression(prog, run_name.c_str()));

    REQ_NVRTC(CudaDynamicLoader::nvrtcCompileProgram(prog, num_compile_flags, compile_flags));

    const char *init_lowered;
    REQ_NVRTC(CudaDynamicLoader::nvrtcGetLoweredName(prog, init_name.c_str(), &init_lowered));
    const char *run_lowered;
    REQ_NVRTC(CudaDynamicLoader::nvrtcGetLoweredName(prog, run_name.c_str(), &run_lowered));

    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(init_out, mod, init_lowered));
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(run_out, mod, run_lowered));

    REQ_NVRTC(CudaDynamicLoader::nvrtcDestroyProgram(&prog));
}

static MegakernelCache loadMegakernelCache(const std::string &cache_path)
{
    std::ifstream cache_file(cache_path,
        std::ios::binary | std::ios::ate);
    if (!cache_file.is_open()) {
        FATAL("Failed to open megakernel cache file at %s",
              cache_path.c_str());
    }

    size_t num_cache_bytes = cache_file.tellg();
    cache_file.seekg(std::ios::beg);
    HeapArray<char> cache_data(num_cache_bytes);
    cache_file.read(cache_data.data(), cache_data.size());

    size_t cur_cache_offset = 0;
    size_t cache_remaining = cache_data.size();

    size_t init_ecs_len = strnlen(cache_data.data() + cur_cache_offset,
                                  cache_remaining);
    if (init_ecs_len == 0 || init_ecs_len == cache_remaining) {
        FATAL("Invalid cache file: no init ecs string");
    }

    const char *init_ecs_str = cache_data.data() + cur_cache_offset;
    cur_cache_offset += init_ecs_len + 1;
    cache_remaining -= init_ecs_len + 1;

    size_t init_worlds_len = strnlen(cache_data.data() + cur_cache_offset,
                                     cache_remaining);

    if (init_worlds_len == 0 || init_worlds_len == cache_remaining) {
        FATAL("Invalid cache_file: no init worlds string");
    }

    const char *init_worlds_str = cache_data.data() + cur_cache_offset;
    cur_cache_offset += init_worlds_len + 1;
    cache_remaining -= init_worlds_len + 1;

    size_t init_tasks_len = strnlen(cache_data.data() + cur_cache_offset,
                                    cache_remaining);

    if (init_tasks_len == 0 || init_tasks_len == cache_remaining) {
        FATAL("Invalid cache file: no kernel string\n");
    }

    const char *init_tasks_str = cache_data.data() + cur_cache_offset;
    cur_cache_offset += init_tasks_len + 1;
    cache_remaining -= init_tasks_len + 1;

    size_t aligned_cubin_offset = utils::roundUpPow2(cur_cache_offset, 4);
    if (aligned_cubin_offset  >= num_cache_bytes) {
        FATAL("Invalid cache file: no CUBIN");
    }

    void *cubin_ptr = cache_data.data() + aligned_cubin_offset;
    size_t num_cubin_bytes = num_cache_bytes - aligned_cubin_offset;

    return MegakernelCache {
        .data = std::move(cache_data),
        .cubinStart = cubin_ptr,
        .numCubinBytes = num_cubin_bytes,
        .initECSName = init_ecs_str,
        .initWorldsName = init_worlds_str,
        .initTasksName = init_tasks_str,
    };
}

static std::string getMegakernelConfigSuffixStr(
    const MegakernelConfig &megakernel_cfg)
{
    return std::to_string(megakernel_cfg.numThreads) + "_" +
        std::to_string(megakernel_cfg.numBlocksPerSM) + "_" +
        std::to_string(megakernel_cfg.numSMs);
}

bool kernelCacheNeedsRecompile(
    const std::string &cache_path,
    const std::vector<std::string> &src_paths)
{
    if (!std::filesystem::exists(cache_path)) {
        return true;
    }
    std::filesystem::file_time_type cache_time = 
        std::filesystem::last_write_time(cache_path);

    // Check if any source file is newer than cache
    for (const auto &src_path : src_paths) {
        if (!std::filesystem::exists(src_path)) {
            continue;
        }

        auto src_time = std::filesystem::last_write_time(src_path);
        if (src_time > cache_time) {
            return true;
        }
    }

    return false;
}

static GPUCompileResults compileCode(
    const char **sources, int64_t num_sources,
    const char **compile_flags, int64_t num_compile_flags,
    const char **fast_compile_flags, int64_t num_fast_compile_flags,
    const char **linker_flags , int64_t num_linker_flags,
    const MegakernelConfig *megakernel_cfgs,
    int64_t num_megakernel_cfgs,
    CompileConfig::OptMode opt_mode,
    ExecutorMode exec_mode, bool verbose_compile)
{
    const std::string cache_path = getenv("MADRONA_MWGPU_KERNEL_CACHE");

    const char *py_root_env = getenv("MADRONA_ROOT_PATH");
    std::filesystem::path root_dir = py_root_env ? py_root_env : std::filesystem::current_path();
    std::vector<std::string> src_paths;
    for (int64_t i = 0; i < num_sources; ++i) {
        src_paths.push_back(std::filesystem::weakly_canonical(root_dir / sources[i]).string());
    }

    bool need_recompile = kernelCacheNeedsRecompile(cache_path, src_paths);
    if (!need_recompile) {
        auto kernel_cache = loadMegakernelCache(cache_path.c_str());

        CUmodule mod;
        REQ_CU(CudaDynamicLoader::cuModuleLoadData(&mod, kernel_cache.cubinStart));
        printf("Loaded megakernel cache from %s\n", cache_path.c_str());

        return {
            .mod = mod,
            .initECSName = kernel_cache.initECSName,
            .initWorldsName = kernel_cache.initWorldsName,
            .initTasksName = kernel_cache.initTasksName,
        };
    }

    nvJitLinkHandle linker;
    REQ_NVJITLINK(nvJitLinkCreate(
        &linker, num_linker_flags, linker_flags));

    auto printLinkerLogs = [&linker](FILE *out) {
        size_t info_log_size, err_log_size;
        REQ_NVJITLINK(
            nvJitLinkGetInfoLogSize(linker, &info_log_size));
        REQ_NVJITLINK(
            nvJitLinkGetErrorLogSize(linker, &err_log_size));

        if (info_log_size > 0) {
            HeapArray<char> info_log(info_log_size);
            REQ_NVJITLINK(nvJitLinkGetInfoLog(linker, info_log.data()));

            fprintf(out, "%s\n", info_log.data());
        }

        if (err_log_size > 0) {
            HeapArray<char> err_log(err_log_size);
            REQ_NVJITLINK(nvJitLinkGetErrorLog(linker, err_log.data()));

            fprintf(out, "%s\n", err_log.data());
        }
    };

    auto checkLinker = [&printLinkerLogs](nvJitLinkResult res) {
        if (res != NVJITLINK_SUCCESS) {
            fprintf(stderr, "CUDA linking Failed!\n");

            printLinkerLogs(stderr);

            fprintf(stderr, "\n");

            ERR_NVJITLINK(res);
        }
    };

#if 0
    // Don't need the device runtime without dynamic parallelism

    checkLinker(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY,
                              MADRONA_CUDADEVRT_PATH,
                              0, nullptr, nullptr));
#endif

    nvJitLinkInputType linker_input_type;
    if (opt_mode == CompileConfig::OptMode::LTO) {
        linker_input_type = NVJITLINK_INPUT_LTOIR;
    } else {
        linker_input_type = NVJITLINK_INPUT_CUBIN;
    }

    auto addToLinker = [&](const HeapArray<char> &cubin, const char *name) {
        checkLinker(nvJitLinkAddData(linker, linker_input_type,
            (char *)cubin.data(), cubin.size(), name));
    };

    std::string megakernel_job_prefix = R"__(#include "megakernel_job_impl.inl"

extern "C" {

)__";

    std::string megakernel_taskgraph_prefix = R"__(#include "megakernel_impl.inl"

extern "C" {

)__";

    std::string megakernel_job_body = R"__(namespace madrona {
namespace mwGPU {

static __attribute__((always_inline)) inline void dispatch(
        uint32_t func_id,
        madrona::JobContainerBase *data,
        uint32_t *data_indices,
        uint32_t *invocation_offsets,
        uint32_t num_launches,
        uint32_t grid)
{
    switch (func_id) {
)__";

    std::string megakernel_taskgraph_body = R"__(namespace madrona {
namespace mwGPU {

static __attribute__((always_inline)) inline void dispatch(
        uint32_t func_id,
        NodeBase *node_data,
        uint32_t invocation_offset)
{
    switch (func_id) {
)__";

    std::string megakernel_prefix;
    std::string megakernel_body;
    std::string megakernel_func_ids;
    std::string_view entry_prefix;
    std::string_view entry_postfix;
    std::string_view entry_params;
    std::string_view entry_args;
    std::string_view id_prefix;
    if (exec_mode == ExecutorMode::JobSystem) {
        megakernel_prefix = megakernel_job_prefix;
        megakernel_body = megakernel_job_body;
        entry_prefix = ".weak .func _ZN7madrona5mwGPU8jobEntry";
        entry_postfix = "EvPNS_16JobContainerBaseEPj";
        entry_params = "(madrona::JobContainerBase *, uint32_t *, uint32_t *, uint32_t, uint32_t);\n";
        entry_args = "(data, data_indices, invocation_offsets, num_launches, grid);\n";
        id_prefix = "_ZN7madrona5mwGPU13JobFuncIDBase";
    } else if (exec_mode == ExecutorMode::TaskGraph) {
        megakernel_prefix = megakernel_taskgraph_prefix;
        megakernel_body = megakernel_taskgraph_body;
        entry_prefix = ".weak .func _ZN7madrona5mwGPU9userEntry";
        entry_postfix = "EvPNS_8NodeBaseEi";
        entry_params = "(madrona::NodeBase *, int32_t);\n";
        entry_args = "(node_data, invocation_offset);\n";
        id_prefix = "_ZN7madrona5mwGPU14UserFuncIDBase";
    }

    std::string init_ecs_name;
    std::string init_worlds_name;
    std::string init_tasks_name;

    uint32_t cur_func_id = 0;
    auto processPTXSymbols = [
            &init_ecs_name, &init_worlds_name, &init_tasks_name,
            &megakernel_prefix, &megakernel_body, &megakernel_func_ids,
            &cur_func_id, entry_prefix, entry_postfix,
            entry_params, entry_args, id_prefix](std::string_view ptx) {

        using namespace std::literals;
        using SizeT = std::string_view::size_type;

        constexpr std::string_view init_ecs_prefix =
            ".entry _ZN7madrona5mwGPU12entryKernels7initECS"sv;
        constexpr std::string_view init_worlds_prefix =
            ".entry _ZN7madrona5mwGPU12entryKernels10initWorlds"sv;
        constexpr std::string_view init_tasks_prefix =
            ".entry _ZN7madrona5mwGPU12entryKernels9initTasks"sv;

        auto findInit = [&ptx](std::string_view prefix, std::string *out) {
            constexpr SizeT init_skip = ".entry "sv.size();

            SizeT prefix_pos = ptx.find(prefix);
            if (prefix_pos != ptx.npos) {
                SizeT start_pos = prefix_pos + init_skip;
                SizeT term_pos = ptx.substr(start_pos).find('(');
                assert(term_pos != ptx.npos);
                *out = ptx.substr(start_pos, term_pos);
            }
        };

        findInit(init_ecs_prefix, &init_ecs_name);
        findInit(init_worlds_prefix, &init_worlds_name);
        findInit(init_tasks_prefix, &init_tasks_name);

        // Search for megakernel entry points
        constexpr SizeT start_skip = ".weak .func "sv.size();

        SizeT cur_pos = 0;
        SizeT found_pos;
        while ((found_pos = ptx.find(entry_prefix, cur_pos)) != ptx.npos) {
            SizeT start_pos = found_pos + start_skip;
            SizeT paren_pos = ptx.find('(', start_pos);
            SizeT endline_pos = ptx.find('\n', start_pos);
            cur_pos = paren_pos;

            // This is a forward declaration so skip it
            if (paren_pos > endline_pos) {
                continue;
            }

            auto mangled_fn = ptx.substr(start_pos, paren_pos - start_pos);
            megakernel_prefix += "void "sv;
            megakernel_prefix += mangled_fn;
            megakernel_prefix += entry_params;

            auto id_str = std::to_string(cur_func_id);

            SizeT postfix_start = mangled_fn.find(entry_postfix);
            assert(postfix_start != mangled_fn.npos);

            SizeT id_common_start = entry_prefix.size() - ".weak .func "sv.size();
            auto common = mangled_fn.substr(id_common_start,
                                            postfix_start - id_common_start);
            
            megakernel_func_ids += "uint32_t ";
            megakernel_func_ids += id_prefix;
            megakernel_func_ids += common;
            megakernel_func_ids += "2idE = ";
            megakernel_func_ids += id_str;
            megakernel_func_ids += ";\n";

            megakernel_body += "        case ";
            megakernel_body += id_str;
            megakernel_body += ": {\n";
            megakernel_body += "            ";
            megakernel_body += mangled_fn;
            megakernel_body += entry_args;
            megakernel_body += "        } break;\n";

            cur_func_id++;
        }
    };

    // Need to save bytecode until nvJitLinkComplete is called,
    // unlike old driver API
    DynArray<HeapArray<char>> source_bytecodes(num_sources);
    if (verbose_compile) {
        printf("Compiling GPU engine code:\n");
    }
    for (int64_t i = 0; i < num_sources; i++) {
        if (verbose_compile) {
            printf("%s\n", src_paths[i].c_str());
        }
        auto [ptx, bytecode] = cu::jitCompileCPPFile(src_paths[i].c_str(),
            compile_flags, num_compile_flags,
            fast_compile_flags, num_fast_compile_flags,
            opt_mode == CompileConfig::OptMode::LTO);

        processPTXSymbols(std::string_view(ptx.data(), ptx.size()));
        addToLinker(bytecode, src_paths[i].c_str());

        source_bytecodes.emplace_back(std::move(bytecode));
    }

    megakernel_prefix += R"__(
}

)__";

    megakernel_body += R"__(        default:
            __builtin_unreachable();
    }
}

}
}

)__";

    HeapArray<const char *> compile_flags_ext(num_compile_flags + 2);
    HeapArray<const char *> fast_compile_flags_ext(
        num_fast_compile_flags + 2);

    for (int64_t i = 0; i < num_compile_flags; i++) {
        compile_flags_ext[i] = compile_flags[i];
    }

    for (int64_t i = 0; i < num_fast_compile_flags; i++) {
        fast_compile_flags_ext[i] = fast_compile_flags[i];
    }

    compile_flags_ext[num_compile_flags] = "-maxrregcount";
    compile_flags_ext[num_compile_flags + 1] = nullptr;

    fast_compile_flags_ext[num_fast_compile_flags] = "-maxrregcount";
    fast_compile_flags_ext[num_fast_compile_flags + 1] = nullptr;

    for (int64_t i = 0; i < num_megakernel_cfgs; i++) {
        const MegakernelConfig &megakernel_cfg = megakernel_cfgs[i];
        std::string megakernel_cfg_suffix =
            getMegakernelConfigSuffixStr(megakernel_cfg);

        std::string megakernel_postfix = "extern \"C\" __global__ void\n";
        megakernel_postfix += "__launch_bounds__(";
        megakernel_postfix += std::to_string(megakernel_cfg.numThreads) +
            ", " + std::to_string(megakernel_cfg.numBlocksPerSM) + ")\n";

        megakernel_postfix += "madronaMWGPUMegakernel_" +
            megakernel_cfg_suffix +
            "(uint32_t taskgraph_id, int32_t start_node_idx, int32_t end_node_idx)\n";

        megakernel_postfix += R"__({
    madrona::mwGPU::megakernelImpl(taskgraph_id, start_node_idx, end_node_idx, )__";
        megakernel_postfix += std::to_string(megakernel_cfg.numBlocksPerSM);

        megakernel_postfix += R"__();
}
)__";

        std::string specialized_megakernel =
            megakernel_prefix + megakernel_body + megakernel_postfix;

        if (i == 0) {
            specialized_megakernel += megakernel_func_ids;
        }

        std::string megakernel_file = "megakernel_" + megakernel_cfg_suffix +
            ".cpp";
        std::string fake_megakernel_cpp_path = std::filesystem::weakly_canonical(
            root_dir / MADRONA_MW_GPU_DEVICE_SRC_DIR / megakernel_file).string();

        uint32_t max_registers =
            65536 / megakernel_cfg.numThreads / megakernel_cfg.numBlocksPerSM;
        // align to multiple of 8
        max_registers -= max_registers % 8;
        max_registers = max_registers > 255 ? 255 : max_registers;
        std::string regcount_str = std::to_string(max_registers);

        compile_flags_ext[num_compile_flags + 1] = regcount_str.c_str();
        fast_compile_flags_ext[num_fast_compile_flags + 1] = regcount_str.c_str();

        auto compiled_megakernel = cu::jitCompileCPPSrc(
            specialized_megakernel.c_str(),
            fake_megakernel_cpp_path.c_str(),
            compile_flags_ext.data(), compile_flags_ext.size(),
            fast_compile_flags_ext.data(), fast_compile_flags_ext.size(),
            opt_mode == CompileConfig::OptMode::LTO);

        addToLinker(compiled_megakernel.outputBinary, megakernel_file.c_str());
    }

    {
        auto *print_megakernel_func_ids = 
            getenv("MADRONA_MWGPU_PRINT_FUNC_IDS");

        if (print_megakernel_func_ids && print_megakernel_func_ids[0] == '1') {
            std::cout << megakernel_func_ids << std::endl;
        }
    }

    checkLinker(nvJitLinkComplete(linker));

    size_t cubin_size;
    REQ_NVJITLINK(nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    HeapArray<char> linked_cubin(cubin_size);
    REQ_NVJITLINK(nvJitLinkGetLinkedCubin(linker, linked_cubin.data()));

    if (!cache_path.empty()) {
        std::ofstream cache_file(cache_path.c_str(), std::ios::binary);
        cache_file.write(init_ecs_name.data(), init_ecs_name.size() + 1);
        cache_file.write(init_worlds_name.data(), init_worlds_name.size() + 1);
        cache_file.write(init_tasks_name.data(), init_tasks_name.size() + 1);
        while (size_t(cache_file.tellp()) % 4 != 0) {
            cache_file.put(0);
        }
        cache_file.write(linked_cubin.data(), linked_cubin.size());

        cache_file.close();
    }

    if (verbose_compile) {
        printf("CUDA linking info:\n");
        printLinkerLogs(stdout);
    }

    CUmodule mod;
    REQ_CU(CudaDynamicLoader::cuModuleLoadData(&mod, linked_cubin.data()));

    REQ_NVJITLINK(nvJitLinkDestroy(&linker));

    if (init_ecs_name.size() == 0 ||
        init_worlds_name.size() == 0 ||
        init_tasks_name.size() == 0) {
        FATAL("Could not find initialization entry points. Make sure to use MADRONA_BUILD_MWGPU_ENTRY somewhere.");
    }
    
    return {
        .mod = mod,
        .initECSName = std::move(init_ecs_name),
        .initWorldsName = std::move(init_worlds_name),
        .initTasksName = std::move(init_tasks_name),
    };
}

template<typename KernelCache>
static KernelCache loadKernelCache(const std::string &cache_path)
{
    std::ifstream cache_file(cache_path,
        std::ios::binary | std::ios::ate);
    if (!cache_file.is_open()) {
        FATAL("Failed to open BVH kernel cache file at %s", cache_path.c_str());
    }

    size_t num_cache_bytes = cache_file.tellg();
    cache_file.seekg(std::ios::beg);
    HeapArray<char> cache_data(num_cache_bytes);
    cache_file.read(cache_data.data(), cache_data.size());

    void *cubin_ptr = cache_data.data();
    size_t num_cubin_bytes = num_cache_bytes;

    return KernelCache {
        .data = std::move(cache_data),
        .cubinStart = cubin_ptr,
        .numCubinBytes = num_cubin_bytes,
    };
}

// We still want to have the same compiler flags as passed to the megakernel
static BVHKernels buildBVHKernels(const CompileConfig &cfg,
                                  int32_t num_sms,
                                  std::pair<int, int> cuda_arch,
                                  CudaBatchRenderConfig::RenderMode render_mode)
{
    using namespace std;

    const std::string cache_path = getenv("MADRONA_BVH_KERNEL_CACHE");

    std::vector<std::string> bvh_srcs = {
            MADRONA_MW_GPU_BVH_INTERNAL_CPP
        };
    const char *py_root_env = getenv("MADRONA_ROOT_PATH");
    std::filesystem::path root_dir = py_root_env ? py_root_env : std::filesystem::current_path();
    std::for_each(
        bvh_srcs.begin(), bvh_srcs.end(),
        [&root_dir](std::string &src) {
            src = std::filesystem::weakly_canonical(root_dir / src).string();
        }
    );

    char *verbose_compile_env = getenv("MADRONA_MWGPU_VERBOSE_COMPILE");
    bool verbose_compile = verbose_compile_env && verbose_compile_env[0] == '1';

    bool need_recompile = kernelCacheNeedsRecompile(
        cache_path,
        bvh_srcs);

    CUmodule mod;
    if (!need_recompile) {
        auto bvh_cache = loadKernelCache<BVHKernelCache>(cache_path);
        REQ_CU(CudaDynamicLoader::cuModuleLoadData(&mod, bvh_cache.cubinStart));
        if (verbose_compile)
        {
            printf("Loaded BVH kernel cache from %s\n", cache_path.c_str());
        }
    }
    else {
        const char *force_debug_env = getenv("MADRONA_MWGPU_FORCE_DEBUG");
        const char *enable_trace_split_env = getenv("MADRONA_TRACK_TRACE_SPLIT");
        const char *shadow_enable_env = getenv("MADRONA_RT_SHADOWS");

        // Build architecture string for this GPU
        string gpu_arch_str = "sm_" + to_string(cuda_arch.first) +
            to_string(cuda_arch.second);

        std::string gpu_arch_flag =
            std::string("-arch=") + gpu_arch_str;

        DynArray<const char *> linker_flags {
            gpu_arch_flag.c_str(),
            "-ftz=1",
            "-prec-div=0",
            "-prec-sqrt=0",
            "-fma=1",
            // "-optimize-unused-variables",
            "-lto",
            "-lineinfo",
            "-maxrregcount=96"
        };

        if (force_debug_env != nullptr && force_debug_env[0] == '1') {
            linker_flags.push_back("-g");
            if (verbose_compile) {
                printf("Compiling with debug\n");
            }
        }

        nvJitLinkHandle linker;
        REQ_NVJITLINK(nvJitLinkCreate(
            &linker, linker_flags.size(), linker_flags.data()));

        nvJitLinkInputType linker_input_type = NVJITLINK_INPUT_LTOIR;

        auto printLinkerLogs = [&linker](FILE *out) {
            size_t info_log_size, err_log_size;
            REQ_NVJITLINK(
                nvJitLinkGetInfoLogSize(linker, &info_log_size));
            REQ_NVJITLINK(
                nvJitLinkGetErrorLogSize(linker, &err_log_size));

            if (info_log_size > 0) {
                HeapArray<char> info_log(info_log_size);
                REQ_NVJITLINK(nvJitLinkGetInfoLog(linker, info_log.data()));

                fprintf(out, "%s\n", info_log.data());
            }

            if (err_log_size > 0) {
                HeapArray<char> err_log(err_log_size);
                REQ_NVJITLINK(nvJitLinkGetErrorLog(linker, err_log.data()));

                fprintf(out, "%s\n", err_log.data());
            }
        };

        auto checkLinker = [&printLinkerLogs](nvJitLinkResult res) {
            if (res != NVJITLINK_SUCCESS) {
                fprintf(stderr, "CUDA linking Failed!\n");

                printLinkerLogs(stderr);

                fprintf(stderr, "\n");

                ERR_NVJITLINK(res);
            }
        };

        auto addToLinker = [&](const HeapArray<char> &cubin, const char *name) {
            checkLinker(nvJitLinkAddData(linker, linker_input_type,
                (char *)cubin.data(), cubin.size(), name));
        };

        std::vector<const char *> common_compile_flags {
            MADRONA_NVRTC_OPTIONS
            "-dlto",
            "-DMADRONA_MWGPU_BVH_MODULE",
            gpu_arch_flag.c_str(),
            "-lineinfo",
            "-maxrregcount=96"
        };
        std::vector<std::string> nvrtc_incl_defs = {
            MADRONA_NVRTC_INCLUDE_DIRS
        };
        std::for_each(
            nvrtc_incl_defs.begin(), nvrtc_incl_defs.end(),
            [&root_dir](std::string &def) {
                def = "-I" + std::filesystem::weakly_canonical(root_dir / def).string();
            }
        );
        for (std::string const & def : nvrtc_incl_defs) {
            common_compile_flags.push_back(def.c_str());
        }

        if (shadow_enable_env && shadow_enable_env[0] == '1') {
            common_compile_flags.push_back("-DMADRONA_RT_SHADOWS");
        }

        if (enable_trace_split_env && enable_trace_split_env[0] == '1') {
            common_compile_flags.push_back("-DMADRONA_PROFILE_BVH_KERNEL");
        }

        if (force_debug_env != nullptr && force_debug_env[0] == '1') {
            common_compile_flags.push_back("--device-debug");
        }

        common_compile_flags.push_back("-DMADRONA_TLAS_WIDTH=8");

        bool render_rgb = (render_mode == CudaBatchRenderConfig::RenderMode::RGBD);

        if (render_rgb) {
            common_compile_flags.push_back("-DMADRONA_RENDER_RGB=1");
        }

        for (const char *user_flag : cfg.userCompileFlags) {
            common_compile_flags.push_back(user_flag);
        }

        DynArray<const char *> fast_compile_flags(common_compile_flags.size());
        for (const char *flag : common_compile_flags) {
            fast_compile_flags.push_back(flag);
        }

        std::vector<const char *> compile_flags = std::move(common_compile_flags);

        for (uint32_t i = 0; i < bvh_srcs.size(); ++i) {
            auto cur_compile_flags = compile_flags;
            std::string cur_path = bvh_srcs[i];

            std::ifstream bvh_file_stream(bvh_srcs[i]);
            std::string bvh_src((std::istreambuf_iterator<char>(bvh_file_stream)),
                std::istreambuf_iterator<char>());

            if (verbose_compile) {
                printf("Compiling %s\n", bvh_srcs[i].c_str());
            }

            // Gives us LTOIR
            auto jit_output = cu::jitCompileCPPSrc(
                bvh_src.c_str(), bvh_srcs[i].c_str(),
                cur_compile_flags.data(), cur_compile_flags.size(),
                cur_compile_flags.data(), cur_compile_flags.size(),
                true);

            addToLinker(jit_output.outputBinary, bvh_srcs[i].c_str());
        }

        checkLinker(nvJitLinkComplete(linker));

        size_t cubin_size;
        REQ_NVJITLINK(nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
        HeapArray<char> linked_cubin(cubin_size);
        REQ_NVJITLINK(nvJitLinkGetLinkedCubin(linker, linked_cubin.data()));

        if (!cache_path.empty()) {
            std::ofstream cache_file(cache_path, std::ios::binary);
            cache_file.write(linked_cubin.data(), linked_cubin.size());
            cache_file.close();
        }

        REQ_CU(CudaDynamicLoader::cuModuleLoadData(&mod, linked_cubin.data()));
        REQ_NVJITLINK(nvJitLinkDestroy(&linker));
    }

    CUfunction bvh_build_fast;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_build_fast, mod,
                               "bvhBuildFast"));

    CUfunction bvh_build_slow;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_build_slow, mod,
                               "bvhBuildSlow"));

    CUfunction bvh_init;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_init, mod,
                               "bvhInit"));

    CUfunction bvh_alloc;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_alloc, mod,
                               "bvhAllocInternalNodes"));

    CUfunction bvh_aabbs;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_aabbs, mod,
                               "bvhConstructAABBs"));

    CUfunction widen_tree;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&widen_tree, mod,
                               "bvhWidenTree"));

    CUfunction bvh_opt;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_opt, mod,
                               "bvhOptimizeLBVH"));

    CUfunction bvh_debug;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_debug, mod,
                               "bvhDebug"));

    CUfunction bvh_raycast_entry;
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&bvh_raycast_entry, mod,
                               "bvhRaycastEntry"));

    CUevent alloc_event;
    CUevent build_event;
    CUevent widen_event;
    CUevent trace_event;
    CUevent stop_event;

    CudaDynamicLoader::cuEventCreate(&alloc_event, CU_EVENT_DEFAULT);
    CudaDynamicLoader::cuEventCreate(&build_event, CU_EVENT_DEFAULT);
    CudaDynamicLoader::cuEventCreate(&widen_event, CU_EVENT_DEFAULT);
    CudaDynamicLoader::cuEventCreate(&trace_event, CU_EVENT_DEFAULT);
    CudaDynamicLoader::cuEventCreate(&stop_event, CU_EVENT_DEFAULT);

    auto *timing_info = (mwGPU::madrona::KernelTimingInfo *)
        cu::allocReadback(sizeof(mwGPU::madrona::KernelTimingInfo));

    BVHKernels bvh_kernels = {
        .numSMs = (uint32_t)num_sms,
        .mod = mod,
        .init = bvh_init,
        .allocInternalNodes = bvh_alloc,
        .buildFast = bvh_build_fast,
        .buildSlow = bvh_build_slow,
        .optFast = bvh_opt,
        .debug = bvh_debug,
        .raycast = bvh_raycast_entry,
        .constructAABBs = bvh_aabbs,
        .widenTree = widen_tree,
        .allocEvent = alloc_event,
        .buildEvent = build_event,
        .widenEvent = widen_event,
        .traceEvent = trace_event,
        .stopEvent = stop_event,
        .recordedTimings = {},
        .timingInfo = timing_info,
        .raycastRGBD = (uint32_t)(render_mode == 
                CudaBatchRenderConfig::RenderMode::RGBD),
        .meshBVHData = {},
        .materialData = {},
    };

    return bvh_kernels;
}

static GPUKernels buildKernels(const CompileConfig &cfg,
                               Span<const MegakernelConfig> megakernel_cfgs,
                               ExecutorMode exec_mode,
                               int32_t num_sms,
                               std::pair<int, int> cuda_arch)
{
    CompileConfig::OptMode opt_mode = cfg.optMode;
    
    const char *force_debug_env = getenv("MADRONA_MWGPU_FORCE_DEBUG");

    if (force_debug_env && force_debug_env[0] == '1') {
        opt_mode = CompileConfig::OptMode::Debug;
    }

    using namespace std;

    array job_sys_cpp_files {
        MADRONA_MW_GPU_JOB_SYS_INTERNAL_CPP
    };

    array task_graph_cpp_files {
        MADRONA_MW_GPU_TASK_GRAPH_INTERNAL_CPP
    };

    uint32_t num_exec_srcs = 0;
    const char **exec_srcs = nullptr;
    if (exec_mode == ExecutorMode::JobSystem) {
        num_exec_srcs = job_sys_cpp_files.size();
        exec_srcs = job_sys_cpp_files.data();
    } else if (exec_mode == ExecutorMode::TaskGraph) {
        num_exec_srcs = task_graph_cpp_files.size();
        exec_srcs = task_graph_cpp_files.data();
    }
    size_t num_srcs = num_exec_srcs + cfg.userSources.size();

    HeapArray<const char *> all_cpp_files(num_srcs);
    memcpy(all_cpp_files.data(), exec_srcs,
           sizeof(const char *) * num_exec_srcs);

    memcpy(all_cpp_files.data() + num_exec_srcs,
           cfg.userSources.data(),
           sizeof(const char *) * cfg.userSources.size());

    // Build architecture string for this GPU
    string gpu_arch_str = "sm_" + to_string(cuda_arch.first) +
        to_string(cuda_arch.second);

    string gpu_arch_flag = std::string("-arch=") + gpu_arch_str;

    string num_sms_str =
        "-DMADRONA_MWGPU_NUM_SMS=(" + to_string(num_sms) + "_i32)";

    CountT max_megakernel_blocks_per_sm = 1;
    for (const MegakernelConfig &megakernel_cfg : megakernel_cfgs) {
        if (megakernel_cfg.numBlocksPerSM > max_megakernel_blocks_per_sm) {
            max_megakernel_blocks_per_sm = megakernel_cfg.numBlocksPerSM;
        }
    }

    string max_blocks_str = "-DMADRONA_MWGPU_MAX_BLOCKS_PER_SM=(" +
        to_string(max_megakernel_blocks_per_sm) + "_i32)";

    DynArray<const char *> common_compile_flags {
        MADRONA_NVRTC_OPTIONS
        gpu_arch_flag.c_str(),
        num_sms_str.c_str(),
        max_blocks_str.c_str(),
#ifdef MADRONA_TRACING
        "-DMADRONA_TRACING=1",
#endif
    };
    std::vector<std::string> nvrtc_incl_defs = {
        MADRONA_NVRTC_INCLUDE_DIRS
    };
    const char *py_root_env = getenv("MADRONA_ROOT_PATH");
    std::filesystem::path root_dir = py_root_env ? py_root_env : std::filesystem::current_path();
    std::for_each(
        nvrtc_incl_defs.begin(), nvrtc_incl_defs.end(),
        [&root_dir](std::string &def) {
            def = "-I" + std::filesystem::weakly_canonical(root_dir / def).string();
        }
    );
    for (std::string const & def : nvrtc_incl_defs) {
        common_compile_flags.push_back(def.c_str());
    }

    for (const char *user_flag : cfg.userCompileFlags) {
        common_compile_flags.push_back(user_flag);
    }

    DynArray<const char *> fast_compile_flags(common_compile_flags.size());
    for (const char *flag : common_compile_flags) {
        fast_compile_flags.push_back(flag);
    }

    DynArray<const char *> compile_flags = std::move(common_compile_flags);

    // No way to disable optimizations in nvrtc besides enabling debug mode
    fast_compile_flags.push_back("-G");

    if (opt_mode == CompileConfig::OptMode::Debug) {
        compile_flags.push_back("--device-debug");
    } else {
        compile_flags.push_back("-dopt=on");
        compile_flags.push_back("--extra-device-vectorization");
        compile_flags.push_back("-lineinfo");
    }

    if (opt_mode == CompileConfig::OptMode::LTO) {
        compile_flags.push_back("-dlto");
        compile_flags.push_back("-DMADRONA_MWGPU_LTO_MODE=1");
    }

    if (exec_mode == ExecutorMode::JobSystem) {
        compile_flags.push_back("-DMARONA_MWGPU_JOB_SYSTEM=1");
    } else if (exec_mode == ExecutorMode::TaskGraph) {
        compile_flags.push_back("-DMADRONA_MWGPU_TASKGRAPH=1");
    }

    DynArray<const char *> linker_flags {
        gpu_arch_flag.c_str(),
        "-ftz=1",
        "-prec-div=0",
        "-prec-sqrt=0",
        "-fma=1",
        "-optimize-unused-variables",
    };

    if (opt_mode == CompileConfig::OptMode::Debug) {
        linker_flags.push_back("-g");
    } else {
        linker_flags.push_back("-lineinfo");
        linker_flags.push_back("-lto");
    }

    char *verbose_compile_env = getenv("MADRONA_MWGPU_VERBOSE_COMPILE");
    bool verbose_compile = verbose_compile_env && verbose_compile_env[0] == '1';

    if (verbose_compile) {
        linker_flags.push_back("-verbose");
    }

    if (verbose_compile) {
        cout << "Compiler Flags:\n";
        for (const char *flag : compile_flags) {
            cout << flag << "\n";
        }

        cout << "\nLinker Flags:\n";
        for (const char *flag : linker_flags) {
            cout << flag << "\n";
        }

        cout << endl;
    }

    auto compile_results = compileCode(
        all_cpp_files.data(), all_cpp_files.size(),
        compile_flags.data(), compile_flags.size(),
        fast_compile_flags.data(), fast_compile_flags.size(),
        linker_flags.data(), linker_flags.size(),
        megakernel_cfgs.data(), megakernel_cfgs.size(),
        opt_mode, exec_mode, verbose_compile);

    HeapArray<CUfunction> megakernel_fns(megakernel_cfgs.size());
    for (int64_t i = 0; i < megakernel_cfgs.size(); i++) {
        const MegakernelConfig &megakernel_cfg = megakernel_cfgs[i];
        std::string kernel_name = "madronaMWGPUMegakernel_" +
            getMegakernelConfigSuffixStr(megakernel_cfg);

        REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&megakernel_fns[i],
                                   compile_results.mod, kernel_name.c_str()));
    }

    GPUKernels gpu_kernels {
        .mod = compile_results.mod,
        .megakernels = std::move(megakernel_fns),
        .computeGPUImplConsts = nullptr,
        .initECS = nullptr,
        .initWorlds = nullptr,
        .initTasks = nullptr,
        .queueUserInit = nullptr,
        .queueUserRun = nullptr,
        .initBVHParams = nullptr,
        .destroyECS = nullptr,
    };

    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.computeGPUImplConsts,
        gpu_kernels.mod, "madronaMWGPUComputeConstants"));

    if (exec_mode == ExecutorMode::JobSystem) {
        REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.initECS, gpu_kernels.mod,
                                   "madronaMWGPUInitialize"));
        // FIXME: getUserEntries is broken
        getUserEntries("", gpu_kernels.mod, compile_flags.data(),
            compile_flags.size(), &gpu_kernels.queueUserInit,
            &gpu_kernels.queueUserRun);
    } else if (exec_mode == ExecutorMode::TaskGraph) {
        REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.initECS, gpu_kernels.mod,
                                   compile_results.initECSName.c_str()));
        REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.initWorlds, gpu_kernels.mod,
                                   compile_results.initWorldsName.c_str()));
        REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.initTasks, gpu_kernels.mod,
                                   compile_results.initTasksName.c_str()));
    }

    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.initBVHParams, gpu_kernels.mod,
                               "initBVHParams"));
    REQ_CU(CudaDynamicLoader::cuModuleGetFunction(&gpu_kernels.destroyECS, gpu_kernels.mod,
                               "freeECSTables"));

    return gpu_kernels;
}

template <typename ...Ts>
HeapArray<void *> makeKernelArgBuffer(Ts ...args)
{
    if constexpr (sizeof...(args) == 0) {
        HeapArray<void *> arg_buffer(6);
        arg_buffer[0] = (void *)uintptr_t(0);
        arg_buffer[1] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        arg_buffer[2] = nullptr;
        arg_buffer[3] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        arg_buffer[4] = &arg_buffer[0];
        arg_buffer[5] = CU_LAUNCH_PARAM_END;

        return arg_buffer;
    } else {
        uint32_t num_arg_bytes = 0;
        auto incrementArgSize = [&num_arg_bytes](auto v) {
            using T = decltype(v);
            num_arg_bytes = utils::roundUp(num_arg_bytes,
                                           (uint32_t)std::alignment_of_v<T>);
            num_arg_bytes += sizeof(T);
        };

        ( incrementArgSize(args), ... );

        auto getArg0Align = [](auto arg0, auto ...) {
            return std::alignment_of_v<decltype(arg0)>;
        };

        uint32_t arg0_alignment = getArg0Align(args...);

        uint32_t total_buf_size = sizeof(void *) * 5;
        uint32_t arg_size_ptr_offset = utils::roundUp(total_buf_size,
            (uint32_t)std::alignment_of_v<size_t>);

        total_buf_size = arg_size_ptr_offset + sizeof(size_t);

        uint32_t arg_ptr_offset =
            utils::roundUp(total_buf_size, arg0_alignment);

        total_buf_size = arg_ptr_offset + num_arg_bytes;

        HeapArray<void *> arg_buffer(utils::divideRoundUp(
                total_buf_size, (uint32_t)sizeof(void *)));

        size_t *arg_size_start = (size_t *)(
            (char *)arg_buffer.data() + arg_size_ptr_offset);

        new (arg_size_start) size_t(num_arg_bytes);

        void *arg_start = (char *)arg_buffer.data() + arg_ptr_offset;

        uint32_t cur_arg_offset = 0;
        auto copyArgs = [arg_start, &cur_arg_offset](auto v) {
            using T = decltype(v);

            cur_arg_offset = utils::roundUp(cur_arg_offset,
                                            (uint32_t)std::alignment_of_v<T>);

            memcpy((char *)arg_start + cur_arg_offset, &v, sizeof(T));

            cur_arg_offset += sizeof(T);
        };

        ( copyArgs(args), ... );

        arg_buffer[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        arg_buffer[1] = arg_start;
        arg_buffer[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        arg_buffer[3] = arg_size_start;
        arg_buffer[4] = CU_LAUNCH_PARAM_END;

        return arg_buffer;
    }
}

static void mapGPUMemory(CUdevice dev, CUdeviceptr base, uint64_t num_bytes)
{
    CUmemAllocationProp alloc_prop {};
    alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop.location.id = dev;

    CUmemGenericAllocationHandle mem;
    REQ_CU(CudaDynamicLoader::cuMemCreate(&mem, num_bytes,
                       &alloc_prop, 0));

    REQ_CU(CudaDynamicLoader::cuMemMap(base, num_bytes, 0, mem, 0));
    REQ_CU(CudaDynamicLoader::cuMemRelease(mem));

    CUmemAccessDesc access_ctrl;
    access_ctrl.location = alloc_prop.location;
    access_ctrl.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    REQ_CU(CudaDynamicLoader::cuMemSetAccess(base, num_bytes,
                          &access_ctrl, 1));

}

static void gpuVMAllocatorThread(HostChannel *channel, CUcontext cu_ctx,
                                 FreeQueue *free_queue)
{
    using namespace std::chrono_literals;
    using cuda::std::memory_order_acquire;
    using cuda::std::memory_order_relaxed;
    using cuda::std::memory_order_release;

    char *verbose_host_alloc_env = getenv("MADRONA_MWGPU_VERBOSE_HOSTALLOC");
    bool verbose_host_alloc =
        verbose_host_alloc_env && verbose_host_alloc_env[0] == '1';

    REQ_CU(CudaDynamicLoader::cuCtxSetCurrent(cu_ctx));

    CUdevice dev;
    REQ_CU(CudaDynamicLoader::cuCtxGetDevice(&dev));

    while (true) {
        while (channel->ready.load(memory_order_acquire) != 1) {
            std::this_thread::sleep_for(1ms);
        }
        channel->ready.store(0, memory_order_relaxed);

        if (channel->op == HostChannel::Op::Reserve) {
            uint64_t num_reserve_bytes = channel->reserve.maxBytes;
            uint64_t num_alloc_bytes = channel->reserve.initNumBytes;

            CUdeviceptr dev_ptr;
            REQ_CU(CudaDynamicLoader::cuMemAddressReserve(&dev_ptr, num_reserve_bytes,
                                       0, 0, 0));

            if (verbose_host_alloc) {
                printf("Reserve request received %lu %lu, got %p\n",
                       num_reserve_bytes, num_alloc_bytes, (void *)dev_ptr);
            }

            if (num_alloc_bytes > 0) {
                mapGPUMemory(dev, dev_ptr, num_alloc_bytes);
            }

            channel->reserve.result = (void *)dev_ptr;
        } else if (channel->op == HostChannel::Op::Map) {
            void *ptr = channel->map.addr;
            uint64_t num_bytes = channel->map.numBytes;

            if (verbose_host_alloc) {
                printf("Grow request received %p %lu\n",
                       ptr, num_bytes);
            }

            mapGPUMemory(dev, (CUdeviceptr)ptr, num_bytes);

            if (verbose_host_alloc) {
                printf("Grew %p\n", ptr);
            }
        } else if (channel->op == HostChannel::Op::Alloc) {
            CUdeviceptr dev_ptr;
            REQ_CU(CudaDynamicLoader::cuMemAlloc(&dev_ptr, channel->alloc.numBytes));
            channel->alloc.result = (void *)dev_ptr;

            if (verbose_host_alloc) {
                printf("Alloc request received %lu, got %p\n",
                    (uint64_t)channel->alloc.numBytes, (void *)dev_ptr);
            }
        } else if (channel->op == HostChannel::Op::ReserveFree) {
            void *ptr = channel->reserveFree.addr;
            uint64_t num_bytes = channel->reserveFree.numBytes;
            uint64_t num_reserve_bytes = channel->reserveFree.numReserveBytes;

            if (verbose_host_alloc) {
                printf("Unmapped %lu bytes from %p, and freed %lu reservation\n",
                    num_bytes, ptr, num_reserve_bytes);
            }

            free_queue->reserveToFree.push_back(FreeQueue::Reserve {
                .addr = channel->reserveFree.addr,
                .numBytes = channel->reserveFree.numBytes,
                .numReserveBytes = channel->reserveFree.numReserveBytes,
            });
        } else if (channel->op == HostChannel::Op::AllocFree) {
            void *ptr = channel->allocFree.addr;

            if (verbose_host_alloc) {
                printf("Allocation free request received for %p\n",
                    ptr);
            }

            free_queue->toFree.push_back(ptr);
        } else if (channel->op == HostChannel::Op::Terminate) {
            break;
        }

        channel->finished.store(1, memory_order_release);
    }
}

static GPUEngineState initEngineAndUserState(
    uint32_t num_worlds,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
    void *world_init_ptr,
    uint32_t num_world_init_bytes,
    void *user_cfg_host_ptr,
    uint32_t num_user_cfg_bytes,
    uint32_t num_taskgraphs,
    uint32_t num_exported,
    const GPUKernels &gpu_kernels,
    const BVHKernels &bvh_kernels,
    const Optional<CudaBatchRenderConfig> &render_cfg,
    ExecutorMode exec_mode,
    CUdevice cu_gpu,
    CUcontext cu_ctx,
    cudaStream_t strm)
{
    CudaDynamicLoader::ensureLoaded();

    int num_sms;
    int shared_mem_per_sm;

    { // Get properties about the GPU
        REQ_CU(CudaDynamicLoader::cuDeviceGetAttribute(
            &num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_gpu));
        REQ_CU(CudaDynamicLoader::cuDeviceGetAttribute(
               &shared_mem_per_sm, 
               CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
               cu_gpu));
    }

    assert(num_taskgraphs > 0);

    auto launchKernel = [strm](CUfunction f, uint32_t num_blocks,
                               uint32_t num_threads,
                               HeapArray<void *> &args) {
        REQ_CU(CudaDynamicLoader::cuLaunchKernel(f, num_blocks, 1, 1, num_threads, 1, 1,
                              0, strm, nullptr, args.data()));
    };

    uint64_t num_init_bytes =
        (uint64_t)num_world_init_bytes * (uint64_t)num_worlds;
    auto init_tmp_buffer = cu::allocGPU(num_init_bytes);
    REQ_CUDA(cudaMemcpyAsync(init_tmp_buffer, world_init_ptr,
        num_init_bytes, cudaMemcpyHostToDevice, strm));

    auto user_cfg_gpu_buffer = cu::allocGPU(num_user_cfg_bytes);
    REQ_CUDA(cudaMemcpyAsync(user_cfg_gpu_buffer, user_cfg_host_ptr,
        num_user_cfg_bytes, cudaMemcpyHostToDevice, strm));

    auto gpu_consts_readback = (GPUImplConsts *)cu::allocReadback(
        sizeof(GPUImplConsts));

    auto gpu_state_size_readback = (size_t *)cu::allocReadback(
        sizeof(size_t));

    auto exported_readback = (void **)cu::allocReadback(
        sizeof(void *) * num_exported);

    CUdeviceptr allocator_channel_devptr;
    REQ_CU(CudaDynamicLoader::cuMemAllocManaged(&allocator_channel_devptr,
                             sizeof(HostChannel), CU_MEM_ATTACH_GLOBAL));
    REQ_CU(CudaDynamicLoader::cuMemAdvise((CUdeviceptr)allocator_channel_devptr, sizeof(HostChannel),
                       CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
    REQ_CU(CudaDynamicLoader::cuMemAdvise((CUdeviceptr)allocator_channel_devptr, sizeof(HostChannel),
                       CU_MEM_ADVISE_SET_ACCESSED_BY, CU_DEVICE_CPU));

    REQ_CU(CudaDynamicLoader::cuMemAdvise(allocator_channel_devptr, sizeof(HostChannel),
                       CU_MEM_ADVISE_SET_ACCESSED_BY, cu_gpu));

    HostChannel *allocator_channel = (HostChannel *)allocator_channel_devptr;

    size_t cu_va_alloc_granularity;
    {
        CUmemAllocationProp default_prop {};
        default_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        default_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        default_prop.location.id = cu_gpu;
        REQ_CU(CudaDynamicLoader::cuMemGetAllocationGranularity(&cu_va_alloc_granularity,
            &default_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    }

    HostAllocInit alloc_init {
        std::max((uint64_t)sysconf(_SC_PAGESIZE),
                 (uint64_t)cu_va_alloc_granularity),
        (uint64_t)cu_va_alloc_granularity,
        allocator_channel,
    };

    FreeQueue *fq = new FreeQueue {};

    std::thread allocator_thread(
        gpuVMAllocatorThread, allocator_channel, cu_ctx, fq);

    auto host_print = std::make_unique<HostPrintCPU>(cu_gpu);

    void *bvh_internals;
    void *bvh_ptrs;
    uint32_t num_bvhs;
    uint32_t raycast_output_width;
    uint32_t raycast_output_height;
    if (render_cfg.has_value()) {
        bvh_ptrs = render_cfg->geoBVHData.meshBVHs;
        num_bvhs = (uint32_t)render_cfg->geoBVHData.numBVHs;

        raycast_output_width = render_cfg->renderWidth;
        raycast_output_height = render_cfg->renderHeight;

        bvh_internals = cu::allocGPU(
            sizeof(mwGPU::madrona::BVHInternalData));
    } else {
        bvh_ptrs = nullptr;
        num_bvhs = 0;

        raycast_output_width = 0;
        raycast_output_height = 0;

        bvh_internals = nullptr;
    }

    auto compute_consts_args = makeKernelArgBuffer(num_worlds,
                                                   num_world_data_bytes,
                                                   world_data_alignment,
                                                   num_taskgraphs,
                                                   gpu_consts_readback,
                                                   gpu_state_size_readback,
                                                   bvh_ptrs,
                                                   num_bvhs,
                                                   raycast_output_width,
                                                   raycast_output_height,
                                                   bvh_internals,
                                                   bvh_kernels.raycastRGBD);

    auto init_ecs_args = makeKernelArgBuffer(alloc_init,
                                             host_print->getChannelPtr(),
                                             exported_readback,
                                             user_cfg_gpu_buffer);

    auto init_tasks_args = makeKernelArgBuffer(
        num_taskgraphs, user_cfg_gpu_buffer);

    auto init_worlds_args = makeKernelArgBuffer(num_worlds,
        user_cfg_gpu_buffer,
        init_tmp_buffer);

    auto no_args = makeKernelArgBuffer();

    launchKernel(gpu_kernels.computeGPUImplConsts, 1, 1,
                 compute_consts_args);

    REQ_CUDA(cudaStreamSynchronize(strm));

    auto gpu_state_buffer = cu::allocGPU(*gpu_state_size_readback);
    cu::deallocCPU(gpu_state_size_readback);

    // The initial values of these pointers are equal to their offsets from
    // the base pointer. Now that we have the base pointer, write the
    // real pointer values.
    gpu_consts_readback->jobSystemAddr =
        (char *)gpu_consts_readback->jobSystemAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->taskGraph =
        (char *)gpu_consts_readback->taskGraph +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->stateManagerAddr =
        (char *)gpu_consts_readback->stateManagerAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->worldDataAddr =
        (char *)gpu_consts_readback->worldDataAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->hostAllocatorAddr =
        (char *)gpu_consts_readback->hostAllocatorAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->hostPrintAddr =
        (char *)gpu_consts_readback->hostPrintAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->tmpAllocatorAddr =
        (char *)gpu_consts_readback->tmpAllocatorAddr +
        (uintptr_t)gpu_state_buffer;

#ifdef MADRONA_TRACING
    gpu_consts_readback->deviceTracingAddr =
        (char *)gpu_consts_readback->deviceTracingAddr +
        (uintptr_t)gpu_state_buffer;

    auto device_tracing = std::make_unique<DeviceTracingManager>(
        gpu_consts_readback->deviceTracingAddr);
#endif

    CUdeviceptr job_sys_consts_addr;
    size_t job_sys_consts_size;
    REQ_CU(CudaDynamicLoader::cuModuleGetGlobal(&job_sys_consts_addr, &job_sys_consts_size,
                             gpu_kernels.mod, "madronaMWGPUConsts"));
    REQ_CU(CudaDynamicLoader::cuMemcpyHtoD(job_sys_consts_addr, gpu_consts_readback,
                        job_sys_consts_size));

    if (exec_mode == ExecutorMode::JobSystem) {
        launchKernel(gpu_kernels.initWorlds, 1, consts::numMegakernelThreads,
                     no_args);
    
        uint32_t num_queue_blocks =
            utils::divideRoundUp(num_worlds, consts::numEntryQueueThreads);

        launchKernel(gpu_kernels.queueUserInit, num_queue_blocks,
                     consts::numEntryQueueThreads, init_worlds_args); 

        launchKernel(gpu_kernels.megakernels[0], 1,
                     consts::numMegakernelThreads, no_args);
    } else if (exec_mode == ExecutorMode::TaskGraph) {
        launchKernel(gpu_kernels.initECS, 1, 1, init_ecs_args);

        uint32_t num_init_blocks =
            utils::divideRoundUp(num_worlds, consts::numMegakernelThreads);

        launchKernel(gpu_kernels.initWorlds, num_init_blocks,
                     consts::numMegakernelThreads, init_worlds_args);
        launchKernel(gpu_kernels.initTasks, 1, 1, init_tasks_args);
    }

    REQ_CUDA(cudaStreamSynchronize(strm));

    cu::deallocGPU(user_cfg_gpu_buffer);
    cu::deallocGPU(init_tmp_buffer);

    HeapArray<void *> exported_cols(num_exported);
    memcpy(exported_cols.data(), exported_readback,
           sizeof(void *) * (uint64_t)num_exported);

    cu::deallocCPU(exported_readback);

    if (render_cfg.has_value()) { 
        auto params_tmp = cu::allocGPU(sizeof(mwGPU::madrona::BVHParams));

        // Address to the BVHParams struct
        CUdeviceptr bvh_consts_addr;
        size_t bvh_consts_size;
        REQ_CU(CudaDynamicLoader::cuModuleGetGlobal(&bvh_consts_addr, &bvh_consts_size,
                                 bvh_kernels.mod, "bvhParams"));

        // Setup the BVH parameters in the __constant__ block of the bvh module
        // Pass this to the ECS to fill in
        auto init_bvh_args = makeKernelArgBuffer(
            (void *)params_tmp,
            num_worlds,
            bvh_internals,
            bvh_ptrs,
            num_bvhs,
            bvh_kernels.timingInfo,
            render_cfg->materialData.materials,
            render_cfg->materialData.textures,
            render_cfg->nearPlane,
            (uint32_t)num_sms,
            (uint32_t)shared_mem_per_sm);

        // Launch the kernel in the megakernel module to initialize the BVH 
        // params
        launchKernel(gpu_kernels.initBVHParams, 1, 1,
                     init_bvh_args);

        REQ_CUDA(cudaStreamSynchronize(strm));

        // Call the bvh init function from bvh module
        auto bvh_init_internal_args = makeKernelArgBuffer(alloc_init);
        launchKernel(bvh_kernels.init, 1, 1,
                     no_args);

        REQ_CUDA(cudaStreamSynchronize(strm));

        CudaDynamicLoader::cuMemcpy(bvh_consts_addr, (CUdeviceptr)params_tmp,
                 sizeof(mwGPU::madrona::BVHParams));

        cu::deallocGPU(params_tmp);
    }

    return GPUEngineState {
        gpu_state_buffer,
        std::move(allocator_thread),
        allocator_channel,
        std::move(host_print),
#ifdef MADRONA_TRACING
        std::move(device_tracing),
#endif
        std::move(exported_cols),
        fq
    };
}

[[maybe_unused]] static CUgraphExec makeJobSysRunGraph(
    CUfunction queue_run_kernel,
    CUfunction job_sys_kernel,
    uint32_t num_worlds)
{
    auto queue_args = makeKernelArgBuffer(num_worlds);
    auto no_args = makeKernelArgBuffer();

    uint32_t num_queue_blocks = utils::divideRoundUp(num_worlds,
        consts::numEntryQueueThreads);

    CUgraph run_graph;
    REQ_CU(CudaDynamicLoader::cuGraphCreate(&run_graph, 0));

    CUDA_KERNEL_NODE_PARAMS kernel_node_params {
        .func = queue_run_kernel,
        .gridDimX = num_queue_blocks,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = consts::numEntryQueueThreads,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .kernelParams = nullptr,
        .extra = queue_args.data(),
        .kern = nullptr,
        .ctx = nullptr,
    };

    CUgraphNode queue_node;
    REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(&queue_node, run_graph,
        nullptr, 0, &kernel_node_params));

    kernel_node_params.func = job_sys_kernel;
    kernel_node_params.gridDimX = 1;
    kernel_node_params.blockDimX = consts::numMegakernelThreads;
    kernel_node_params.extra = no_args.data();

    CUgraphNode job_sys_node;
    REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(&job_sys_node, run_graph,
        &queue_node, 1, &kernel_node_params));

    CUgraphExec run_graph_exec;
    REQ_CU(CudaDynamicLoader::cuGraphInstantiate(&run_graph_exec, run_graph, 0));

    REQ_CU(CudaDynamicLoader::cuGraphDestroy(run_graph));

    return run_graph_exec;
}

static MegakernelConfig processExecConfigOverride(const char *override_str)
{
    auto err = []() {
        FATAL("MADRONA_MWGPU_CONFIG_OVERRIDE format invalid");
    };

    const char *comma1 = strchr(override_str, ',');
    if (comma1 == nullptr) err();

    uint32_t num_threads_override;
    auto res1 = std::from_chars(override_str, comma1, num_threads_override);
    if (res1.ec != std::errc {}) err();

    override_str = comma1 + 1;

    const char *comma2 = strchr(override_str, ',');
    if (comma2 == nullptr) err();

    uint32_t num_blocks_override;
    auto res2 = std::from_chars(override_str, comma2, num_blocks_override);
    if (res2.ec != std::errc {}) err();

    override_str = comma2 + 1;
    const char *end = override_str + strlen(override_str);

    uint32_t num_sms_override;
    auto res3 = std::from_chars(override_str, end, num_sms_override);
    if (res3.ec != std::errc {}) err();

    return MegakernelConfig {
        num_threads_override,
        num_blocks_override,
        num_sms_override,
    };
}

static int64_t findMatchingMegakernelConfig(
    const MegakernelConfig *megakernel_cfgs,
    int64_t num_megakernels,
    const MegakernelConfig &desired_cfg)
{
    for (int64_t i = 0; i < num_megakernels; i++) {
        const MegakernelConfig &cur_cfg = megakernel_cfgs[i];

        if (desired_cfg.numThreads == cur_cfg.numThreads &&
                desired_cfg.numBlocksPerSM == cur_cfg.numBlocksPerSM &&
                desired_cfg.numSMs == cur_cfg.numSMs) {
            return i;
        }
    }

    FATAL("Could not find megakernel configuration %u %u %u",
          desired_cfg.numThreads, desired_cfg.numBlocksPerSM,
          desired_cfg.numSMs);
};

static DynArray<int32_t> processExecConfigFile(
    const char *file_path,
    int64_t default_megakernel_idx,
    const MegakernelConfig *megakernel_cfgs,
    int64_t num_configs)
{
    using namespace simdjson;

    char *verbose_taskgraph_env = getenv("MADRONA_MWGPU_VERBOSE_TASKGRAPH");
    bool verbose_taskgraph =
        verbose_taskgraph_env && verbose_taskgraph_env[0] == '1';

    padded_string json_data;
    REQ_JSON(padded_string::load(file_path).get(json_data));

    ondemand::parser parser;
    ondemand::document config_json_doc;
    REQ_JSON(parser.iterate(json_data).get(config_json_doc));

    ondemand::object config_json_obj;
    REQ_JSON(config_json_doc.get(config_json_obj));

    DynArray<int32_t> node_megakernels(0);

    for (auto kv : config_json_obj) {
        std::string_view key;
        REQ_JSON(kv.unescaped_key().get(key));
        uint64_t num_blocks;
        REQ_JSON(kv.value().get(num_blocks));

        uint64_t node_idx;
        auto res = std::from_chars(key.data(), key.data() + key.size(),
                                   node_idx);
        
        if (res.ec != std::errc {}) {
            FATAL("MADRONA_MWGPU_EXEC_CONFIG_FILE points to invalid file");
        }

        // FIXME: json only has desired # blocks, somewhat ambiguous
        MegakernelConfig node_cfg {
            megakernel_cfgs[0].numThreads,
            (uint32_t)num_blocks,
            megakernel_cfgs[0].numSMs,
        };

        int64_t megakernel_idx = findMatchingMegakernelConfig(
            megakernel_cfgs, num_configs, node_cfg);

        if (verbose_taskgraph)
        {
            printf("Taskgraph node %lu: using config %d %d %d\n", node_idx,
                node_cfg.numThreads, node_cfg.numBlocksPerSM, node_cfg.numSMs);
        }

        if ((CountT)node_idx >= node_megakernels.size()) {
            node_megakernels.resize(node_idx + 1, [&](int32_t *v) {
                *v = default_megakernel_idx;
            });
        }

        node_megakernels[node_idx] = megakernel_idx;
    }

    return node_megakernels;
}

static CUgraphExec makeTaskGraphRunGraph(
    Span<const uint32_t> taskgraph_ids,
    const MegakernelConfig *megakernel_cfgs,
    const CUfunction *megakernels,
    int64_t num_megakernels,
    const char *stat_name,
    CUevent *start, CUevent *end)
{
    CUgraph run_graph;
    REQ_CU(CudaDynamicLoader::cuGraphCreate(&run_graph, 0));

    char *config_override_env = getenv("MADRONA_MWGPU_EXEC_CONFIG_OVERRIDE");
    char *config_file_env = getenv("MADRONA_MWGPU_EXEC_CONFIG_FILE");

    int64_t default_megakernel_idx = 0;

    if (config_override_env != nullptr) {
        MegakernelConfig override_cfg =
            processExecConfigOverride(config_override_env);

        default_megakernel_idx = findMatchingMegakernelConfig(
            megakernel_cfgs, num_megakernels, override_cfg);
    }

    DynArray<int32_t> node_megakernels = config_file_env != nullptr ?
        processExecConfigFile(config_file_env,
                              default_megakernel_idx,
                              megakernel_cfgs,
                              num_megakernels) :
        DynArray<int32_t>({(int32_t)default_megakernel_idx});

    DynArray<CUgraphNode> megakernel_launches(0);

    if (stat_name) {
        CUgraphNode start_node;
        REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(&start_node,
                    run_graph, nullptr, 0, *start));
        megakernel_launches.push_back(start_node);
    }

    auto addMegakernelNode = [&](int64_t megakernel_idx,
                                 CUgraphNode *dependencies,
                                 unsigned int num_dependencies,
                                 void **args) {
        const MegakernelConfig &megakernel_cfg =
            megakernel_cfgs[megakernel_idx];

        CUDA_KERNEL_NODE_PARAMS kernel_node_params {
            .func = megakernels[megakernel_idx],
            .gridDimX = megakernel_cfg.numSMs * megakernel_cfg.numBlocksPerSM,
            .gridDimY = 1,
            .gridDimZ = 1,
            .blockDimX = megakernel_cfg.numThreads,
            .blockDimY = 1,
            .blockDimZ = 1,
            .sharedMemBytes = 0,
            .kernelParams = nullptr,
            .extra = args,
            .kern = nullptr,
            .ctx = nullptr,
        };

        CUgraphNode megakernel_node;
        REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(&megakernel_node, run_graph,
            dependencies, num_dependencies, &kernel_node_params));
        megakernel_launches.push_back(megakernel_node);
    };

    auto addNodesForTaskGraph = [&](uint32_t taskgraph_id) {
        int64_t cur_node_idx = 0;
        while (cur_node_idx < node_megakernels.size()) {
            int64_t cur_megakernel_idx = node_megakernels[cur_node_idx];

            int64_t switch_node_idx;
            for (switch_node_idx = cur_node_idx + 1;
                 switch_node_idx < node_megakernels.size() &&
                     node_megakernels[switch_node_idx] == cur_megakernel_idx;
                 switch_node_idx++) {}

            // FIXME: this use of -1 is hacky. The profiling code should
            // just write the total number of nodes into the json file.
            int64_t end_node_idx = switch_node_idx == node_megakernels.size() ?
                -1 : switch_node_idx;

            CUgraphNode *deps = nullptr;
            unsigned int num_deps = 0;
            if (megakernel_launches.size() > 0) {
                deps = &megakernel_launches.back();
                num_deps = 1;
            }

            auto megakernel_args = makeKernelArgBuffer(
                taskgraph_id, (int32_t)cur_node_idx, (int32_t)end_node_idx);

            addMegakernelNode(cur_megakernel_idx, deps, num_deps,
                              megakernel_args.data());

            cur_node_idx = switch_node_idx;
        }
    };

    for (uint32_t taskgraph_id : taskgraph_ids) {
        addNodesForTaskGraph(taskgraph_id);
    }

    if (stat_name) {
        CUgraphNode end_node;
        REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(&end_node,
                    run_graph, &megakernel_launches.back(), 1,
                    *end));
    }

    CUgraphExec run_graph_exec;
    REQ_CU(CudaDynamicLoader::cuGraphInstantiate(&run_graph_exec, run_graph, 0));

    REQ_CU(CudaDynamicLoader::cuGraphDestroy(run_graph));

    return run_graph_exec;
}

MWCudaLaunchGraph::MWCudaLaunchGraph()
    : impl_(nullptr)
{}

MWCudaLaunchGraph::MWCudaLaunchGraph(Impl *impl)
    : impl_(impl)
{}


MWCudaLaunchGraph::MWCudaLaunchGraph(MWCudaLaunchGraph &&) = default;
MWCudaLaunchGraph & MWCudaLaunchGraph::operator=(MWCudaLaunchGraph &&) = default;

MWCudaLaunchGraph::~MWCudaLaunchGraph()
{
    if (!impl_) {
        return;
    }

    REQ_CU(CudaDynamicLoader::cuGraphExecDestroy(impl_->runGraph));
}

CUcontext MWCudaExecutor::initCUDA(int gpu_id)
{
    CudaDynamicLoader::ensureLoaded();

    REQ_CUDA(cudaSetDevice(gpu_id));
    CUdevice cu_dev;
    REQ_CU(CudaDynamicLoader::cuDeviceGet(&cu_dev, gpu_id));
    CUcontext cu_ctx;
    REQ_CU(CudaDynamicLoader::cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    setCudaHeapSize();
    REQ_CU(CudaDynamicLoader::cuCtxSetCurrent(cu_ctx));
    return cu_ctx;
}

MWCudaExecutor::MWCudaExecutor()
    : impl_(nullptr)
{}

MWCudaExecutor::MWCudaExecutor(
        const StateConfig &state_cfg,
        const CompileConfig &compile_cfg,
        CUcontext cu_ctx,
        const Optional<CudaBatchRenderConfig> &render_cfg)
    : impl_(nullptr)
{
    // Setup CUDA cache directory
    const char *root_cache_env = getenv("MADRONA_ROOT_CACHE_DIR");
    std::filesystem::path root_cache_dir = root_cache_env ? root_cache_env : "/tmp/madrona_cache";
    if(!std::filesystem::exists(root_cache_dir)) {
        std::filesystem::create_directories(root_cache_dir);
    }
    std::string kernel_cache_path = (root_cache_dir / "kernel_cache").string();
    std::string bvh_cache_path = (root_cache_dir / "bvh_cache").string();
    setenv("MADRONA_MWGPU_KERNEL_CACHE", kernel_cache_path.c_str(), 1);
    setenv("MADRONA_BVH_KERNEL_CACHE", bvh_cache_path.c_str(), 1);
    // printf("Kernel cache directory set to: %s\n", kernel_cache_path.c_str());
    // printf("BVH cache directory set to: %s\n", bvh_cache_path.c_str());

    const ExecutorMode exec_mode = ExecutorMode::TaskGraph;

    auto strm = cu::makeStream();

    CUdevice cu_gpu;
    REQ_CU(CudaDynamicLoader::cuCtxGetDevice(&cu_gpu));

    int num_sms;
    REQ_CU(CudaDynamicLoader::cuDeviceGetAttribute(
        &num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_gpu));

    CountT max_megakernel_blocks_per_sm = 1;
    const char *enable_pgo_env = getenv("MADRONA_MWGPU_ENABLE_PGO");
    if (enable_pgo_env && enable_pgo_env[0] == '1') {
        max_megakernel_blocks_per_sm = 6;
    }

    HeapArray<MegakernelConfig> megakernel_cfgs(max_megakernel_blocks_per_sm);
    for (uint32_t i = 0; i < max_megakernel_blocks_per_sm; i++) {
        megakernel_cfgs[i] = {
            consts::numMegakernelThreads,
            i + 1,
            (uint32_t)num_sms,
        };
    }

    std::pair<int, int> cu_capability;
    REQ_CU(CudaDynamicLoader::cuDeviceGetAttribute(&cu_capability.first,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_gpu));
    REQ_CU(CudaDynamicLoader::cuDeviceGetAttribute(&cu_capability.second,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_gpu));

    int shared_mem_per_sm;
    REQ_CU(CudaDynamicLoader::cuDeviceGetAttribute(
           &shared_mem_per_sm, 
           CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
           cu_gpu));

    BVHKernels bvh_kernels {};
    if (render_cfg.has_value()) {
        bvh_kernels = buildBVHKernels(
            compile_cfg, num_sms, cu_capability, render_cfg->renderMode);
        bvh_kernels.meshBVHData = render_cfg->geoBVHData;
        bvh_kernels.materialData = render_cfg->materialData;
    } 

    GPUKernels gpu_kernels = buildKernels(compile_cfg, megakernel_cfgs,
        exec_mode, num_sms, cu_capability);

    GPUEngineState eng_state = initEngineAndUserState(
        state_cfg.numWorlds, state_cfg.numWorldDataBytes,
        state_cfg.worldDataAlignment, state_cfg.worldInitPtr,
        state_cfg.numWorldInitBytes, state_cfg.userConfigPtr,
        state_cfg.numUserConfigBytes,  state_cfg.numTaskGraphs,
        state_cfg.numExportedBuffers,
        gpu_kernels,
        bvh_kernels, render_cfg,
        exec_mode, cu_gpu, cu_ctx, strm);

    TaskGraphsState taskgraphs_state {
        .megakernels = std::move(gpu_kernels.megakernels),
        .megakernelConfigs = std::move(megakernel_cfgs),
        .numTaskgraphs = state_cfg.numTaskGraphs,
    };

    impl_ = std::unique_ptr<Impl>(new Impl {
        strm,
        gpu_kernels.mod,
        std::move(eng_state),
        std::move(taskgraphs_state),
        render_cfg.has_value(),
        bvh_kernels,
        gpu_kernels.destroyECS,
        state_cfg.numWorlds,
        render_cfg.has_value() ? render_cfg->renderWidth : 0,
        render_cfg.has_value() ? render_cfg->renderHeight : 0,
        (uint32_t)shared_mem_per_sm,
        {}
    });

    std::cout << "Initialization finished" << std::endl;
}

MWCudaExecutor::MWCudaExecutor(MWCudaExecutor &&o)
    = default;

MWCudaExecutor & MWCudaExecutor::operator=(MWCudaExecutor &&) = default;

MWCudaExecutor::~MWCudaExecutor()
{
    if (!impl_) return;

    REQ_CU(CudaDynamicLoader::cuLaunchKernel(impl_->destroyKernel, 1, 1, 1, 1, 1, 1,
            0, impl_->cuStream, nullptr, nullptr));
    REQ_CUDA(cudaStreamSynchronize(impl_->cuStream));

    // Ok now, we go through the free queue
    auto *fq = impl_->engineState.freeQueue;
    for (int i = 0; i < (int)fq->toFree.size(); ++i) {
        REQ_CU(CudaDynamicLoader::cuMemFree((CUdeviceptr)fq->toFree[i]));
    }

    for (int i = 0; i < (int)fq->reserveToFree.size(); ++i) {
        void *ptr = fq->reserveToFree[i].addr;
        uint64_t num_reserve_bytes = fq->reserveToFree[i].numReserveBytes;

        REQ_CU(CudaDynamicLoader::cuMemUnmap((CUdeviceptr)ptr, num_reserve_bytes));
        REQ_CU(CudaDynamicLoader::cuMemAddressFree((CUdeviceptr)ptr, num_reserve_bytes));
    }

    // Free mesh bvh data
    if (impl_->bvhKernels.meshBVHData.numNodes) {
        REQ_CUDA(cudaFree(impl_->bvhKernels.meshBVHData.nodes));
    }

    if (impl_->bvhKernels.meshBVHData.numLeaves) {
        REQ_CUDA(cudaFree(impl_->bvhKernels.meshBVHData.leafMaterial));
    }

    if (impl_->bvhKernels.meshBVHData.numVerts) {
        REQ_CUDA(cudaFree(impl_->bvhKernels.meshBVHData.vertices));
    }

    if (impl_->bvhKernels.meshBVHData.numBVHs) {
        REQ_CUDA(cudaFree(impl_->bvhKernels.meshBVHData.meshBVHs));
    }

    if (impl_->bvhKernels.materialData.textures) {
        REQ_CUDA(cudaFree(impl_->bvhKernels.materialData.textures));
    }

    if (impl_->bvhKernels.materialData.materials) {
        REQ_CUDA(cudaFree(impl_->bvhKernels.materialData.materials));
    }

#ifdef MADRONA_TRACING
    // Seems good to copy the logs before the module is unloaded
    impl_->engineState.deviceTracing.reset();
#endif

    impl_->engineState.hostAllocatorChannel->op =
        HostChannel::Op::Terminate;
    impl_->engineState.hostAllocatorChannel->ready.store(
        1, cuda::std::memory_order_release);
    impl_->engineState.allocatorThread.join();

    REQ_CU(CudaDynamicLoader::cuModuleUnload(impl_->cuModule));
    REQ_CUDA(cudaStreamDestroy(impl_->cuStream));

    char *verbose_stats_env = getenv("MADRONA_MWGPU_VERBOSE_STAT");
    bool verbose_stats = verbose_stats_env && verbose_stats_env[0] == '1';
    if (verbose_stats)
    {
        if (impl_->enableRaycasting) { // RT times
            float avg_total_time = 0.f;
            u32 num_times = impl_->bvhKernels.recordedTimings.size();
            for (u32 i = 0; i < num_times; ++i) {
                avg_total_time += impl_->bvhKernels.recordedTimings[i].totalTime /
                    (float)num_times;
            }

            printf("rt avg total time = %f ms\n", avg_total_time);
        }

        // Other times
        for (u32 g = 0; g < impl_->timingGroups.size(); ++g) {
            TimingGroup &times = impl_->timingGroups[g];

            float avg_total_time = 0.f;
            for (u32 i = 0; i < times.recordedTimings.size(); ++i) {
                avg_total_time += times.recordedTimings[i] /
                    (float)times.recordedTimings.size();
            }

            printf("%s avg total time = %f ms\n", times.statName,
                    avg_total_time);
        }
    }
}

MWCudaLaunchGraph MWCudaExecutor::buildRenderGraph()
{
    if (!impl_->enableRaycasting) {
        FATAL("Trying to build render graph when raytracing was disabled.");
    }

    auto &bvh_kernels = impl_->bvhKernels;

    CUgraph run_graph;
    REQ_CU(CudaDynamicLoader::cuGraphCreate(&run_graph, 0));

    DynArray<CUgraphNode> nodes(0);

    auto get_prev_node = [&nodes]() -> CUgraphNode * {
        if (nodes.size() <= 1) {
            return nullptr;
        } else {
            return &nodes[nodes.size() - 2];
        }
    };

    auto get_new_node = [&nodes]() {
        return &nodes[nodes.size() - 1];
    };

    auto push_new_node = [&nodes]() {
        nodes.push_back(CUgraphNode{});
    };

    push_new_node();

    REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(
                get_new_node(), run_graph,
                get_prev_node(), 0,
                bvh_kernels.allocEvent));

    CUDA_KERNEL_NODE_PARAMS bvh_launch_params = {
        .func = bvh_kernels.allocInternalNodes,
        // We want to max-out the GPU for all the BVH kernels
        .gridDimX = 1,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 1,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .kernelParams = nullptr,
        .extra = nullptr,
        .kern = nullptr,
        .ctx = nullptr,
    };

    // Allocation node
    push_new_node();
    REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(get_new_node(), run_graph,
                get_prev_node(), 1,
                &bvh_launch_params));

    bool fast_bvh = true;

    auto shared_mem_per_sm = impl_->sharedMemPerSM;

    if (fast_bvh) {
        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(
                    get_new_node(), run_graph,
                    get_prev_node(), 1,
                    bvh_kernels.buildEvent));

        // Fast LBVH build node
        const uint32_t num_blocks_per_sm_fast_build = 16;
        bvh_launch_params.func = bvh_kernels.buildFast;
        bvh_launch_params.gridDimX = bvh_kernels.numSMs *
            num_blocks_per_sm_fast_build;
        bvh_launch_params.blockDimX = 256;
        bvh_launch_params.sharedMemBytes = shared_mem_per_sm / 
            num_blocks_per_sm_fast_build;

        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(get_new_node(), run_graph,
                    get_prev_node(), 1, 
                    &bvh_launch_params));

        bvh_launch_params.func = bvh_kernels.constructAABBs;

        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(get_new_node(), run_graph,
                    get_prev_node(), 1, 
                    &bvh_launch_params));
    } else {
        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(
                    get_new_node(), run_graph,
                    get_prev_node(), 1,
                    bvh_kernels.buildEvent));

        const uint32_t num_blocks_per_sm_slow_build = 16;
        bvh_launch_params.func = bvh_kernels.buildSlow;
        bvh_launch_params.gridDimX = bvh_kernels.numSMs *
            num_blocks_per_sm_slow_build;
        bvh_launch_params.blockDimX = 256;
        bvh_launch_params.sharedMemBytes = shared_mem_per_sm /
            num_blocks_per_sm_slow_build;

        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(get_new_node(), run_graph,
                    get_prev_node(), 1, &bvh_launch_params));
    }

    { // Push the widen node
        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(
                    get_new_node(), run_graph,
                    get_prev_node(), 1,
                    bvh_kernels.widenEvent));

        const uint32_t num_blocks_per_sm_slow_build = 16;
        bvh_launch_params.func = bvh_kernels.widenTree;
        bvh_launch_params.gridDimX = bvh_kernels.numSMs *
            num_blocks_per_sm_slow_build;
        bvh_launch_params.blockDimX = 256;
        bvh_launch_params.sharedMemBytes = shared_mem_per_sm /
            num_blocks_per_sm_slow_build;

        push_new_node();
        REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(get_new_node(), run_graph,
                    get_prev_node(), 1, &bvh_launch_params));
    }

    // We assign a 4x4 region of blocks per image/view
    // Each block processes 16x16 pixels and we have one thread per pixel.
    push_new_node();
    REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(
                get_new_node(), run_graph,
                get_prev_node(), 1,
                bvh_kernels.traceEvent));

    const uint32_t num_resident_views_per_sm = 4;

    const uint32_t pixels_per_block = 8;

    uint32_t grid_dimY = impl_->renderOutputWidth / pixels_per_block;
    uint32_t grid_dimZ = impl_->renderOutputHeight / pixels_per_block;
    CUDA_KERNEL_NODE_PARAMS bvh_launch_raycast = {
        .func = bvh_kernels.raycast,
        .gridDimX = bvh_kernels.numSMs * num_resident_views_per_sm,
        .gridDimY = grid_dimY,
        .gridDimZ = grid_dimZ,
        .blockDimX = pixels_per_block,
        .blockDimY = pixels_per_block,
        .blockDimZ = 1,
        .sharedMemBytes = 1024,
        .kernelParams = nullptr,
        .extra = nullptr,
        .kern = nullptr,
        .ctx = nullptr,
    };

    push_new_node();
    REQ_CU(CudaDynamicLoader::cuGraphAddKernelNode(get_new_node(), run_graph,
                get_prev_node(), 1,
                &bvh_launch_raycast));

    push_new_node();
    REQ_CU(CudaDynamicLoader::cuGraphAddEventRecordNode(
                get_new_node(), run_graph,
                get_prev_node(), 1,
                bvh_kernels.stopEvent));

    CUgraphExec run_graph_exec;
    REQ_CU(CudaDynamicLoader::cuGraphInstantiate(&run_graph_exec, run_graph, 0));

    REQ_CU(CudaDynamicLoader::cuGraphDestroy(run_graph));

    return MWCudaLaunchGraph(new MWCudaLaunchGraph::Impl {
        .runGraph = run_graph_exec,
        .enableRaytracing = true,
        .start = {},
        .end = {},
        .statName = nullptr,
        .timingGroupIndex = 0,
    });
}

MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(
        Span<const uint32_t> taskgraph_ids,
        const char *stat_name)
{
    CUevent start, end;

    if (stat_name) {
        CudaDynamicLoader::cuEventCreate(&start, CU_EVENT_DEFAULT);
        CudaDynamicLoader::cuEventCreate(&end, CU_EVENT_DEFAULT);
    }

    auto run_graph = makeTaskGraphRunGraph(
        taskgraph_ids,
        impl_->taskGraphsState.megakernelConfigs.data(),
        impl_->taskGraphsState.megakernels.data(),
        impl_->taskGraphsState.megakernels.size(),
        stat_name,
        &start, &end);

    uint32_t timing_group_idx = impl_->timingGroups.size();
    if (stat_name) {
        impl_->timingGroups.push_back({stat_name, {}});
    }

    return MWCudaLaunchGraph(new MWCudaLaunchGraph::Impl {
        .runGraph = run_graph,
        .enableRaytracing = false,
        .start = start,
        .end = end,
        .statName = stat_name,
        .timingGroupIndex = timing_group_idx,
    });
}

MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraphAllTaskGraphs()
{
    const CountT num_taskgraphs = impl_->taskGraphsState.numTaskgraphs;
    HeapArray<uint32_t> all_taskgraph_ids(num_taskgraphs);
    for (CountT i = 0; i < num_taskgraphs; i++) {
        all_taskgraph_ids[i] = i;
    }

    return buildLaunchGraph(all_taskgraph_ids);
}

void MWCudaExecutor::run(MWCudaLaunchGraph &launch_graph)
{
    REQ_CU(CudaDynamicLoader::cuGraphLaunch(launch_graph.impl_->runGraph, impl_->cuStream));
    REQ_CUDA(cudaStreamSynchronize(impl_->cuStream));
#ifdef MADRONA_TRACING
    impl_->engineState.deviceTracing->transferLogToCPU();
#endif

    if (launch_graph.impl_->statName) {
        REQ_CU(CudaDynamicLoader::cuEventSynchronize(launch_graph.impl_->start));
        REQ_CU(CudaDynamicLoader::cuEventSynchronize(launch_graph.impl_->end));

        float sort_ms = 0.f;
        REQ_CU(CudaDynamicLoader::cuEventElapsedTime(&sort_ms,
                    launch_graph.impl_->start, 
                    launch_graph.impl_->end));

        impl_->timingGroups[launch_graph.impl_->timingGroupIndex].
            recordedTimings.push_back(sort_ms);
    } else if (launch_graph.impl_->enableRaytracing) {
        REQ_CU(CudaDynamicLoader::cuEventSynchronize(impl_->bvhKernels.allocEvent));
        REQ_CU(CudaDynamicLoader::cuEventSynchronize(impl_->bvhKernels.stopEvent));

        float total_time = 0.f;
        REQ_CU(CudaDynamicLoader::cuEventElapsedTime(&total_time,
                    impl_->bvhKernels.allocEvent,
                    impl_->bvhKernels.stopEvent));

        impl_->bvhKernels.recordedTimings.push_back({
                .totalTime = total_time,
                .buildTimeRatio = {},
                .traceTimeRatio = {},
                .numTLASTraces = {},
                .numBLASTraces = {},
                .tlasRatio = {},
                .memoryUsage = {},
            });
    }
}

void MWCudaExecutor::runAsync(MWCudaLaunchGraph &launch_graph,
                              cudaStream_t strm)
{
    REQ_CU(CudaDynamicLoader::cuGraphLaunch(launch_graph.impl_->runGraph, strm));
}

void * MWCudaExecutor::getExported(CountT slot) const
{
    return impl_->engineState.exportedColumns[slot];
}

}
