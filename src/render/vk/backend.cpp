#include <madrona/render/vk/backend.hpp>

#include <madrona/macros.hpp>
#include <madrona/utils.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>

#include "config.hpp"
#include "utils.hpp"

#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#if defined(MADRONA_LINUX) || defined(MADRONA_MACOS)
#include <dlfcn.h>
#include <csignal>
#include <filesystem>
#include <cstdio>
#if defined(MADRONA_LINUX)
#include <sys/ptrace.h>
#elif defined(MADRONA_MACOS)
#include <sys/ptrace.h>
#endif
#elif defined(MADRONA_WINDOWS)
#include <windows.h>
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

using namespace std;

namespace madrona::render::vk {

namespace {

struct InitializationDispatch {
    PFN_vkGetInstanceProcAddr
        getInstanceAddr;
    PFN_vkEnumerateInstanceVersion
        enumerateInstanceVersion;
    PFN_vkEnumerateInstanceExtensionProperties
        enumerateInstanceExtensionProperties;
    PFN_vkEnumerateInstanceLayerProperties
        enumerateInstanceLayerProperties;
    PFN_vkCreateInstance createInstance;
};

struct QueueFamilyChoices {
    uint32_t gfxQF;
    uint32_t numGFXQueues;
    uint32_t computeQF;
    uint32_t numComputeQueues;
    uint32_t transferQF;
    uint32_t numTransferQueues;
};

}

struct Backend::Init {
    VkInstance hdl;
    InitializationDispatch dt;
    bool validationEnabled;

    static inline Backend::Init init(PFN_vkGetInstanceProcAddr get_inst_addr,
                                     bool want_validation,
                                     Span<const char *const> extra_exts);
};

LoaderLib::LoaderLib(void *lib, void (*entry_fn)(), const char *env_str)
    : APILib {},
      lib_(lib),
      entry_fn_(entry_fn),
      env_str_(env_str)
{}

LoaderLib::~LoaderLib()
{
    if (!lib_) {
        return;
    }

    free((void *)env_str_);

#if defined(MADRONA_LINUX) || defined(MADRONA_MACOS)
    dlclose(lib_);
#endif
}

LoaderLib * LoaderLib::load()
{
    void *lib = nullptr;
    const char *icd_env = nullptr;

#if defined(MADRONA_LINUX)
    lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        lib = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
    }
#elif defined(MADRONA_MACOS)
    lib = dlopen("libvulkan.1.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        Dl_info dl_info;
        if (!dladdr((void *)&LoaderLib::load, &dl_info)) {
            FATAL("Couldn't find path to libvulkan.1.dylib");
        }

        auto vk_dir = 
            std::filesystem::path(dl_info.dli_fname).parent_path() / "vk";
        auto libvk_path = vk_dir / "libvulkan.1.dylib";
        auto icd_path = vk_dir / "MoltenVK_icd.json";

        icd_env = strdup(icd_path.c_str());
        setenv("VK_ICD_FILENAMES", icd_env, 0);

        lib = dlopen(libvk_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    }
#endif
    if (!lib) {
        FATAL("Couldn't find vulkan loader");
    }

    void (*entry_fn)() = nullptr;
#if defined(MADRONA_LINUX) || defined(MADRONA_MACOS)
    auto get_inst_addr = (PFN_vkGetInstanceProcAddr)dlsym(lib,
        "vkGetInstanceProcAddr");
    if (get_inst_addr == nullptr) {
        FATAL("Couldn't find vkGetInstanceProcAddr");
    }
    
    get_inst_addr = (PFN_vkGetInstanceProcAddr)get_inst_addr(
        VK_NULL_HANDLE, "vkGetInstanceProcAddr");
    if (get_inst_addr == VK_NULL_HANDLE) {
        FATAL("Refetching vkGetInstanceProcAddr after dlsym failed");
    }

    entry_fn = (void (*)())get_inst_addr;
#endif

    return new LoaderLib(lib, entry_fn, icd_env);
}

LoaderLib * LoaderLib::external(void (*entry_fn)())
{
    return new LoaderLib(nullptr, entry_fn, nullptr);
}

static InitializationDispatch fetchInitDispatchTable(
    PFN_vkGetInstanceProcAddr get_inst_addr)
{
    auto get_addr = [&](const char *name) {
        auto ptr = get_inst_addr(nullptr, name);

        if (!ptr) {
            FATAL("Failed to load %s for vulkan initialization.", name);
        }

        return ptr;
    };

    return InitializationDispatch {
        get_inst_addr,
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            get_addr("vkEnumerateInstanceVersion")),
        reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            get_addr("vkEnumerateInstanceExtensionProperties")),
        reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            get_addr("vkEnumerateInstanceLayerProperties")),
        reinterpret_cast<PFN_vkCreateInstance>(get_addr("vkCreateInstance")),
    };
}

static std::string lexSeverityFlags(VkDebugUtilsMessageSeverityFlagsEXT severity) {
    std::string result;

    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        result += "VERBOSE | ";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        result += "INFO | ";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        result += "WARNING | ";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        result += "ERROR | ";

    if (!result.empty())
        result.erase(result.size() - 3);  // Remove trailing " | "

    return result.empty() ? "UNKNOWN" : result;
}

static std::string lexMessageTypeFlags(VkDebugUtilsMessageTypeFlagsEXT types) {
    std::string result;

    if (types & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
        result += "GENERAL | ";
    if (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
        result += "VALIDATION | ";
    if (types & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
        result += "PERFORMANCE | ";

    if (!result.empty())
        result.erase(result.size() - 3);  // Remove trailing " | "

    return result.empty() ? "UNKNOWN" : result;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "[" << lexSeverityFlags(messageSeverity) << " | "
              << lexMessageTypeFlags(messageType) << "] "
              << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

static bool checkValidationAvailable(const InitializationDispatch &dt)
{
    uint32_t num_layers;
    REQ_VK(dt.enumerateInstanceLayerProperties(&num_layers, nullptr));

    HeapArray<VkLayerProperties> layers(num_layers);

    REQ_VK(dt.enumerateInstanceLayerProperties(&num_layers, layers.data()));

    bool have_validation_layer = false;
    for (int layer_idx = 0; layer_idx < (int)num_layers; layer_idx++) {
        const auto &layer_prop = layers[layer_idx];
        if (!strcmp("VK_LAYER_KHRONOS_validation", layer_prop.layerName)) {
            have_validation_layer = true;
            break;
        }
    }

    // FIXME check for VK_EXT_debug_utils

    uint32_t num_exts;
    REQ_VK(dt.enumerateInstanceExtensionProperties(nullptr, &num_exts,
                                                   nullptr));

    HeapArray<VkExtensionProperties> exts(num_exts);

    REQ_VK(dt.enumerateInstanceExtensionProperties(nullptr, &num_exts,
                                                   exts.data()));

    bool have_debug_ext = false;
    for (int ext_idx = 0; ext_idx < (int)num_exts; ext_idx++) {
        const auto &ext_prop = exts[ext_idx];
        if (!strcmp("VK_EXT_debug_utils", ext_prop.extensionName)) {
            have_debug_ext = true;
            break;
        }
    }

    if (have_validation_layer && have_debug_ext) {
        return true;
    } else {
        fprintf(stderr, "Validation layers unavailable\n");
        return false;
    }
}

static bool checkVulkanLayerSupport(const InitializationDispatch &dt, const std::vector<const char*>& requested) {
    uint32_t layerCount;
    REQ_VK(dt.enumerateInstanceLayerProperties(&layerCount, nullptr));
    std::vector<VkLayerProperties> available(layerCount);
    REQ_VK(dt.enumerateInstanceLayerProperties(&layerCount, available.data()));

    for (const char* name : requested) {
        bool found = false;
        for (const auto& layer : available) {
            if (strcmp(name, layer.layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Missing Vulkan layer: " << name << std::endl;
            return false;
        }
    }
    std::cout << "All Vulkan layers supported" << std::endl;
    return true;
}

Backend::Init Backend::Init::init(
    PFN_vkGetInstanceProcAddr get_inst_addr,
    bool want_validation,
    Span<const char *const> extra_exts)
{
    InitializationDispatch dt = fetchInitDispatchTable(get_inst_addr);

    uint32_t inst_version;
    REQ_VK(dt.enumerateInstanceVersion(&inst_version));
    if (VK_API_VERSION_MAJOR(inst_version) == 1 &&
        VK_API_VERSION_MINOR(inst_version) < 2) {
        FATAL("At least Vulkan 1.2 required");
    }

    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "madrona";
    app_info.pEngineName = "madrona";
    app_info.apiVersion = VK_API_VERSION_1_3;

    vector<const char *> layers;
    DynArray<const char *> extensions(extra_exts.size());

    for (const char *extra_ext : extra_exts) {
        extensions.push_back(extra_ext);
    }

    vector<VkValidationFeatureEnableEXT> val_enabled;
    VkValidationFeaturesEXT val_features {};
    val_features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;

    bool enable_validation = want_validation && checkValidationAvailable(dt);

    if (enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        extensions.push_back("VK_KHR_xcb_surface");
        extensions.push_back("VK_KHR_surface");

        val_enabled.push_back(
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);

        char *best_practices = getenv("VK_BEST_VALIDATE");
        if (best_practices && best_practices[0] == '1') {
            val_enabled.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
        }

        char *gpu_debug_test = getenv("VK_GPU_VALIDATE");
        if (gpu_debug_test && gpu_debug_test[0] == '1') {
            val_enabled.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
        } else {
            val_enabled.push_back(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
#if defined(MADRONA_WINDOWS)
            SetEnvironmentVariable("DEBUG_PRINTF_TO_STDOUT", "1");
#else
            setenv("DEBUG_PRINTF_TO_STDOUT", "1", 1);
#endif
        }

        val_features.enabledValidationFeatureCount = val_enabled.size();
        val_features.pEnabledValidationFeatures = val_enabled.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        populateDebugMessengerCreateInfo(debugCreateInfo);
        val_features.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    }

    VkInstanceCreateInfo inst_info {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = &val_features;
    inst_info.pApplicationInfo = &app_info;

#if defined(MADRONA_MACOS) || defined(MADRONA_IOS)
    inst_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    checkVulkanLayerSupport(dt, layers);
    if (layers.size() > 0) {
        inst_info.enabledLayerCount = layers.size();
        inst_info.ppEnabledLayerNames = layers.data();
    }

    if (extensions.size() > 0) {
        inst_info.enabledExtensionCount = extensions.size();
        inst_info.ppEnabledExtensionNames = extensions.data();
    }

    VkInstance inst;
    REQ_VK(dt.createInstance(&inst_info, nullptr, &inst));

    return {
        inst,
        dt,
        enable_validation,
    };
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
validationDebug(VkDebugUtilsMessageSeverityFlagBitsEXT,
                VkDebugUtilsMessageTypeFlagsEXT,
                const VkDebugUtilsMessengerCallbackDataEXT *data,
                void *)
{
    fprintf(stderr, "%s\n", data->pMessage);

#if defined(MADRONA_LINUX)
    // Check if debugger is attached by trying to read from /proc/self/status
    bool debugger_attached = false;
    FILE* status_file = fopen("/proc/self/status", "r");
    if (status_file) {
        char line[256];
        while (fgets(line, sizeof(line), status_file)) {
            if (strncmp(line, "TracerPid:", 10) == 0) {
                int tracer_pid;
                if (sscanf(line + 10, "%d", &tracer_pid) == 1 && tracer_pid != 0) {
                    debugger_attached = true;
                }
                break;
            }
        }
        fclose(status_file);
    }
    
    if (debugger_attached) {
        signal(SIGTRAP, SIG_IGN);
        raise(SIGTRAP);
        signal(SIGTRAP, SIG_DFL);
    }
#elif defined(MADRONA_MACOS)
    // On macOS, check if debugger is attached by checking if we can trace ourselves
    bool debugger_attached = false;
    if (ptrace(PT_TRACE_ME, 0, 0, 0) == 0) {
        // We can trace ourselves, so no debugger is attached
        ptrace(PT_DETACH, 0, 0, 0);
    } else {
        // We can't trace ourselves, so a debugger is attached
        debugger_attached = true;
    }
    
    if (debugger_attached) {
        signal(SIGTRAP, SIG_IGN);
        raise(SIGTRAP);
        signal(SIGTRAP, SIG_DFL);
    }
#elif defined(MADRONA_WINDOWS)
    if (IsDebuggerPresent()) {
        DebugBreak();
    }
#endif

    return VK_FALSE;
}

static VkDebugUtilsMessengerEXT makeDebugCallback(VkInstance hdl,
                                                  PFN_vkGetInstanceProcAddr get_addr)
{
    auto makeMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        get_addr(hdl, "vkCreateDebugUtilsMessengerEXT"));

    assert(makeMessenger != nullptr);

    VkDebugUtilsMessengerEXT messenger;

    VkDebugUtilsMessengerCreateInfoEXT create_info {};
    create_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = validationDebug;

    REQ_VK(makeMessenger(hdl, &create_info, nullptr, &messenger));

    return messenger;
}

Backend::Backend(void (*vk_entry_fn)(),
                 bool enable_validation,
                 bool enable_present,
                 Span<const char *const> extra_exts)
    : Backend(Backend::Init::init((PFN_vkGetInstanceProcAddr)vk_entry_fn,
        enable_validation, extra_exts), enable_present)
{}

Backend::Backend(Init init, bool enable_present)
    : APIBackend {},
      hdl(init.hdl),
      dt(hdl, init.dt.getInstanceAddr, enable_present),
      debug_(init.validationEnabled ?
                makeDebugCallback(hdl, init.dt.getInstanceAddr) :
                VK_NULL_HANDLE)
{}

Backend::Backend(Backend &&o)
    : APIBackend {},
      hdl(o.hdl),
      dt(std::move(o.dt)),
      debug_(std::move(o.debug_))
{
    o.hdl = VK_NULL_HANDLE;
}

Backend::~Backend()
{
    if (hdl == VK_NULL_HANDLE) {
        return;
    }

    if (debug_ != VK_NULL_HANDLE) {
        auto destroy_messenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                dt.getInstanceProcAddr(hdl,
                                       "vkDestroyDebugUtilsMessengerEXT"));
        destroy_messenger(hdl, debug_, nullptr);
    }
    dt.destroyInstance(hdl, nullptr);
}

static void fillQueueInfo(VkDeviceQueueCreateInfo &info,
                          uint32_t idx,
                          const vector<float> &priorities)
{
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.queueFamilyIndex = idx;
    info.queueCount = priorities.size();
    info.pQueuePriorities = priorities.data();
}

VkPhysicalDevice Backend::findPhysicalDevice(
    const DeviceID &dev_id) const
{
    uint32_t num_gpus;
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, nullptr));

    HeapArray<VkPhysicalDevice> phys(num_gpus);
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, phys.data()));

    for (uint32_t idx = 0; idx < phys.size(); idx++) {
        VkPhysicalDevice phy = phys[idx];
        VkPhysicalDeviceIDProperties vk_id_props {};
        vk_id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

        VkPhysicalDeviceProperties2 props {};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &vk_id_props;
        dt.getPhysicalDeviceProperties2(phy, &props);

        if (!memcmp(dev_id.data(), vk_id_props.deviceUUID,
                    sizeof(DeviceID::value_type) * dev_id.size())) {
            return phy;
        }
    }

    FATAL("Cannot find matching vulkan UUID for GPU");
}

static QueueFamilyChoices chooseQueueFamilies(
    VkQueueFamilyProperties2 *queue_families,
    const uint32_t num_queue_families,
    VkPhysicalDevice phy,
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR surface_support_fn,
    Span<const VkSurfaceKHR> present_surfaces)
{
    auto present_check = [&](uint32_t qf_idx) {
        bool supports_present = true;

        for (VkSurfaceKHR surface : present_surfaces) {
            VkBool32 supported;
            REQ_VK(surface_support_fn(
                phy, qf_idx, surface, &supported));

            if (supported == VK_FALSE) {
                supports_present = false;
                break;
            }
        }

        return supports_present;
    };

    constexpr uint32_t qf_sentinel = 0xFFFF'FFFF;

    QueueFamilyChoices res {
        .gfxQF = qf_sentinel,
        .numGFXQueues = 0,
        .computeQF = qf_sentinel,
        .numComputeQueues = 0,
        .transferQF = qf_sentinel,
        .numTransferQueues = 0,
    };

    // Currently only finds dedicated transfer, compute, and gfx queues
    // FIXME implement more flexiblity in queue choices
    // FIXME: allow choosing graphics or compute present support

    // Pick graphics family that can present with largest number of queues
    for (uint32_t i = 0; i < num_queue_families; i++) {
        const auto &qf_prop = queue_families[i].queueFamilyProperties;

        if ((qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) && present_check(i)) {
            if (qf_prop.queueCount > res.numGFXQueues) {
                res.gfxQF = i;
                res.numGFXQueues = qf_prop.queueCount;
            }
        }
    }

    bool dedicated_compute_found = false;
    for (uint32_t i = 0; i < num_queue_families; i++) {
        if (i == res.gfxQF) {
            continue;
        }

        const auto &qf_prop = queue_families[i].queueFamilyProperties;

        if (!(qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            continue;
        }

        bool is_dedicated = !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT);
        if ((is_dedicated && !dedicated_compute_found) ||
                (is_dedicated == dedicated_compute_found &&
                 qf_prop.queueCount > res.numComputeQueues)) {
            res.computeQF = i;
            res.numComputeQueues = qf_prop.queueCount;
        } 
        dedicated_compute_found |= is_dedicated;
    }

    bool dedicated_transfer_found = false;
    for (uint32_t i = 0; i < num_queue_families; i++) {
        if (i == res.gfxQF || i == res.computeQF) {
            continue;
        }

        const auto &qf_prop = queue_families[i].queueFamilyProperties;

        if (!(qf_prop.queueFlags & VK_QUEUE_TRANSFER_BIT)) {
            continue;
        }

        bool is_dedicated = !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT);

        if ((is_dedicated && !dedicated_transfer_found) ||
                (is_dedicated == dedicated_transfer_found &&
                 qf_prop.queueCount > res.numTransferQueues)) {
            res.transferQF = i;
            res.numTransferQueues = qf_prop.queueCount;
        } 
        dedicated_transfer_found |= is_dedicated;
    }

    if (res.gfxQF == qf_sentinel) {
        FATAL("Could not find graphics queue family");
    }

    return res;
}

Device * Backend::makeDevice(
    CountT gpu_idx,
    Span<const VkSurfaceKHR> present_surfaces)
{
#ifdef MADRONA_CUDA_SUPPORT
    DeviceID vk_id;

    int device_count;
    REQ_CUDA(cudaGetDeviceCount(&device_count));
    if ((CountT)device_count <= gpu_idx) {
        FATAL("%ld is not a valid CUDA ID given %d supported device\n",
              gpu_idx, device_count);
    }

    cudaDeviceProp props;
    REQ_CUDA(cudaGetDeviceProperties(&props, (int)gpu_idx));

    if (props.computeMode == cudaComputeModeProhibited) {
        FATAL("%ld corresponds to a prohibited device\n", gpu_idx);
    }

    memcpy(vk_id.data(), &props.uuid,
           sizeof(DeviceID::value_type) * vk_id.size());

    return makeDevice(vk_id, present_surfaces);
#else
    uint32_t num_gpus;
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, nullptr));

    HeapArray<VkPhysicalDevice> phys(num_gpus);
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, phys.data()));

    if (gpu_idx >= num_gpus) {
        FATAL("Requested GPU %u, only %u GPUs detected by Vulkan",
              (uint32_t)gpu_idx, num_gpus);
    }

    return makeDevice(phys[gpu_idx], present_surfaces);
#endif
}

Device * Backend::makeDevice(
    const DeviceID &gpu_id,
    Span<const VkSurfaceKHR> present_surfaces)
{
    VkPhysicalDevice phy = findPhysicalDevice(gpu_id);

    return makeDevice(phy, present_surfaces);
}

Device * Backend::makeDevice(
    VkPhysicalDevice phy,
    Span<const VkSurfaceKHR> present_surfaces)
{
    // FIXME:
    const uint32_t desired_gfx_queues = 2;
    const uint32_t desired_compute_queues = 2;
    const uint32_t desired_transfer_queues = 2;


    DynArray<const char *> extensions {
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
#if defined(MADRONA_MACOS) || defined(MADRONA_IOS)
        "VK_KHR_portability_subset"
#endif
    };

    uint32_t num_supported_extensions;
    REQ_VK(dt.enumerateDeviceExtensionProperties(phy, nullptr,
        &num_supported_extensions, nullptr));

    HeapArray<VkExtensionProperties> supported_extensions(
        num_supported_extensions);
    REQ_VK(dt.enumerateDeviceExtensionProperties(phy, nullptr,
        &num_supported_extensions, supported_extensions.data()));

    bool supports_rt = true;
    {
        bool accel_struct_ext_available = false;
        bool ray_query_ext_available = false;
        for (const VkExtensionProperties &ext : supported_extensions) {
            if (!strcmp(ext.extensionName,
                    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
                accel_struct_ext_available = true;
            } else if (!strcmp(ext.extensionName,
                    VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
                ray_query_ext_available = true;
            }
        }

        supports_rt = accel_struct_ext_available && ray_query_ext_available;
    }

    {
        char *disable_rt_env = getenv("MADRONA_VK_DISABLE_RT");
        if (disable_rt_env && disable_rt_env[0] == '1') {
            supports_rt = false;
        }
    }

    if (supports_rt) {
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }

#if defined(MADRONA_LINUX) && defined(MADRONA_CUDA_SUPPORT)
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    bool supports_mem_export = true;
#else
    bool supports_mem_export = false;
#endif

    if (present_surfaces.size() > 0) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    if (debug_ != VK_NULL_HANDLE) {
#if !defined(MADRONA_MACOS) && !defined(MADRONA_IOS)
        extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
#endif
    }

    VkPhysicalDeviceFeatures2 feats;
    feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats.pNext = nullptr;
    dt.getPhysicalDeviceFeatures2(phy, &feats);

    uint32_t num_queue_families;
    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                               nullptr);

    if (num_queue_families == 0) {
        FATAL("GPU doesn't have any queue families");
    }

    HeapArray<VkQueueFamilyProperties2> queue_family_props(num_queue_families);
    for (auto &qf : queue_family_props) {
        qf.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        qf.pNext = nullptr;
    }

    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                               queue_family_props.data());

    QueueFamilyChoices qf_choices = chooseQueueFamilies(
        queue_family_props.data(), num_queue_families,
        phy, dt.getPhysicalDeviceSurfaceSupportKHR, present_surfaces);

    const uint32_t num_gfx_queues =
        min(desired_gfx_queues, qf_choices.numGFXQueues);
    const uint32_t num_compute_queues =
        min(desired_compute_queues, qf_choices.numComputeQueues);
    const uint32_t num_transfer_queues =
        min(desired_transfer_queues, qf_choices.numTransferQueues);

    array<VkDeviceQueueCreateInfo, 3> queue_infos {};
    vector<float> gfx_pris(num_gfx_queues, VulkanConfig::gfx_priority);
    vector<float> compute_pris(num_compute_queues,
                               VulkanConfig::compute_priority);
    vector<float> transfer_pris(num_transfer_queues,
                                VulkanConfig::transfer_priority);

    uint32_t num_qf_allocated = 0;
    fillQueueInfo(queue_infos[num_qf_allocated++],
                  qf_choices.gfxQF, gfx_pris);

    if (qf_choices.numComputeQueues > 0) {
        fillQueueInfo(queue_infos[num_qf_allocated++],
                      qf_choices.computeQF, compute_pris);
    }

    if (qf_choices.numTransferQueues > 0) {
        fillQueueInfo(queue_infos[num_qf_allocated++],
                      qf_choices.transferQF, transfer_pris);
    }

    VkDeviceCreateInfo dev_create_info {};
    dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_create_info.queueCreateInfoCount = num_qf_allocated;
    dev_create_info.pQueueCreateInfos = queue_infos.data();
    dev_create_info.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    dev_create_info.ppEnabledExtensionNames = extensions.data();

    dev_create_info.pEnabledFeatures = nullptr;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features {};
    accel_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel_features.pNext = nullptr;
    accel_features.accelerationStructure = true;

    VkPhysicalDeviceRayQueryFeaturesKHR rq_features {};
    rq_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq_features.pNext = &accel_features;
    rq_features.rayQuery = true;

#if 0
    VkPhysicalDeviceRobustness2FeaturesEXT robustness_features {};
    robustness_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
    robustness_features.pNext = &rq_features;
    robustness_features.nullDescriptor = true;

    VkPhysicalDeviceLineRasterizationFeaturesEXT line_features {};
    line_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT;
    line_features.pNext = &robustness_features;
    line_features.smoothLines = true;
#endif

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features {};
    atomic_float_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
#if 0
    atomic_float_features.pNext = &line_features;
#endif
    if (supports_rt) {
        atomic_float_features.pNext = &rq_features;
    } else {
        atomic_float_features.pNext = nullptr;
    }
    atomic_float_features.shaderSharedFloat32Atomics = true;
    atomic_float_features.shaderSharedFloat32AtomicAdd = true;

    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_features {};
    subgroup_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup_features.pNext = &atomic_float_features;
    subgroup_features.computeFullSubgroups = true;
    subgroup_features.subgroupSizeControl = true;

    VkPhysicalDeviceDynamicRenderingFeatures dyn_render_features {};
    dyn_render_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dyn_render_features.pNext = &subgroup_features;
    dyn_render_features.dynamicRendering = true;

#if 0
    VkPhysicalDeviceVulkan13Features vk13_features {};
    vk13_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vk13_features.pNext = &atomic_float_features;
    vk13_features.synchronization2 = true;
    vk13_features.computeFullSubgroups = true;
    vk13_features.subgroupSizeControl = true;
#endif

    VkPhysicalDeviceVulkan12Features vk12_features {};
    vk12_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12_features.pNext = &dyn_render_features;
    vk12_features.bufferDeviceAddress = true;
    vk12_features.descriptorIndexing = true;
    vk12_features.descriptorBindingPartiallyBound = true;
    vk12_features.descriptorBindingUpdateUnusedWhilePending = true;
#ifdef MADRONA_MACOS
    vk12_features.drawIndirectCount = false; // No MoltenVK support :(
#else
    vk12_features.drawIndirectCount = true;
#endif
    vk12_features.runtimeDescriptorArray = true;
    vk12_features.shaderStorageBufferArrayNonUniformIndexing = false;
    vk12_features.shaderSampledImageArrayNonUniformIndexing = true;
    vk12_features.shaderFloat16 = true;
    vk12_features.shaderInt8 = true;
    vk12_features.storageBuffer8BitAccess = true;
    vk12_features.shaderOutputLayer = true;
    vk12_features.shaderOutputViewportIndex = true;

    VkPhysicalDeviceVulkan11Features vk11_features {};
    vk11_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vk11_features.pNext = &vk12_features;
    vk11_features.storageBuffer16BitAccess = true;

    VkPhysicalDeviceFeatures2 requested_features {};
    requested_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    requested_features.pNext = &vk11_features;

    requested_features.features.samplerAnisotropy = true;
    requested_features.features.shaderInt16 = true;
    requested_features.features.shaderInt64 = true;
    requested_features.features.wideLines = false; // No MoltenVK support :(
    requested_features.features.fillModeNonSolid = true;
    requested_features.features.multiDrawIndirect = true;
    requested_features.features.multiViewport = true;

#ifdef MADRONA_MACOS
    requested_features.features.geometryShader = false;
#else
    // Required for batch renderer (accessing primitive ID)
    requested_features.features.geometryShader = true;
#endif

    dev_create_info.pNext = &requested_features;

    VkDevice dev;
    REQ_VK(dt.createDevice(phy, &dev_create_info, nullptr, &dev));

    PFN_vkGetDeviceProcAddr get_dev_addr = 
        (PFN_vkGetDeviceProcAddr)dt.getInstanceProcAddr(hdl, "vkGetDeviceProcAddr");
    if (get_dev_addr == VK_NULL_HANDLE) {
        FATAL("Failed to load vkGetDeviceProcAddr");
    }

    VkPhysicalDeviceProperties physical_device_properties;
    dt.getPhysicalDeviceProperties(phy, &physical_device_properties);

    return new Device(
        qf_choices.gfxQF,
        qf_choices.computeQF,
        qf_choices.transferQF,
        num_gfx_queues,
        num_compute_queues,
        num_transfer_queues,
        supports_rt,
        physical_device_properties.limits.maxImageArrayLayers,
        physical_device_properties.limits.maxViewports,
        physical_device_properties.limits.maxImageDimension2D,
        // 1,
        physical_device_properties.limits.timestampPeriod,
        phy,
        dev,
        DeviceDispatch(dev, get_dev_addr,
                       present_surfaces.size() > 0,
                       supports_rt, supports_mem_export));
}

}
