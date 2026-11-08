// G2 Turbo Engine v0.2 — Android Vulkan compute GEMV runtime.
//
// Runs the Q4_0 shader on the actual mobile GPU; NOT a synthetic mock.
// Pixel 7 Pro uses a unified-memory architecture. We prefer coherent,
// host-visible storage buffers, but support noncoherent memory with explicit
// flush/invalidate. All Vk objects are cleaned up on error or success.
//
// The measured GPU time is a host-observed dispatch + fence latency, not an
// on-chip GPU timestamp. It excludes initial upload, pipeline initialization,
// shader compilation, and readback (reported separately where applicable).
#include "vulkan_bench.h"

#include <android/asset_manager.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace g2 {
namespace {

using Clock = std::chrono::steady_clock;

double millis(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

void check(VkResult result, const char* where) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(where) + " VkResult=" +
                                 std::to_string(static_cast<int>(result)));
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[c >> 4] << hex[c & 15];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::vector<uint32_t> shader_spirv(AAssetManager* manager, const std::string& filename) {
    if (!manager) throw std::runtime_error("Android asset manager not supplied");
    AAsset* asset = AAssetManager_open(manager, filename.c_str(),
                                      AASSET_MODE_BUFFER);
    if (!asset)
        throw std::runtime_error("SPIR-V shader asset missing");
    struct Close { void operator()(AAsset* a) const { AAsset_close(a); } };
    std::unique_ptr<AAsset, Close> handle(asset);
    const off_t len = AAsset_getLength(asset);
    if (len <= 0 || len % 4 != 0 || len > 1024 * 1024)
        throw std::runtime_error("Invalid SPIR-V asset size");
    std::vector<uint32_t> words(static_cast<size_t>(len / 4));
    const int bytes_read = AAsset_read(asset, words.data(),
                                       static_cast<size_t>(len));
    if (bytes_read != len || words[0] != 0x07230203u)
        throw std::runtime_error("Could not read SPIR-V module");
    return words;
}

struct GpuBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize length = 0;
    bool coherent = false;
    void* mapped = nullptr;

    GpuBuffer() = default;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    ~GpuBuffer() {
        if (device != VK_NULL_HANDLE) {
            if (mapped) vkUnmapMemory(device, memory);
            if (buffer) vkDestroyBuffer(device, buffer, nullptr);
            if (memory) vkFreeMemory(device, memory, nullptr);
        }
    }

    void create(VkDevice dev, const VkPhysicalDeviceMemoryProperties& props,
                VkDeviceSize bytes) {
        device = dev;
        length = bytes;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &info, nullptr, &buffer),
              "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, buffer, &req);
        int best = -1;
        int best_score = -1;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) == 0) continue;
            auto flags = props.memoryTypes[i].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
            const int score =
                ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 4 : 0) +
                ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 2 : 0) +
                ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
            if (score > best_score) {
                best_score = score;
                best = static_cast<int>(i);
            }
        }
        if (best == -1) {
            throw std::runtime_error("No HOST_VISIBLE Vulkan storage-buffer memory type");
        }
        coherent = (props.memoryTypes[best].propertyFlags &
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = static_cast<uint32_t>(best);
        check(vkAllocateMemory(device, &alloc, nullptr, &memory),
              "vkAllocateMemory");
        check(vkBindBufferMemory(device, buffer, memory, 0),
              "vkBindBufferMemory");
        check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped),
              "vkMapMemory");
    }

    // VK_WHOLE_SIZE flush handles the nonCoherentAtomSize alignment rule.
    void flush() {
        if (coherent) return;
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        check(vkFlushMappedMemoryRanges(device, 1, &range),
              "vkFlushMappedMemoryRanges");
    }

    void invalidate() {
        if (coherent) return;
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        check(vkInvalidateMappedMemoryRanges(device, 1, &range),
              "vkInvalidateMappedMemoryRanges");
    }

    void write(const void* source, size_t bytes) {
        if (bytes > length) throw std::runtime_error("GPU buffer overflow");
        std::memcpy(mapped, source, bytes);
        flush();
    }
};

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    uint32_t timestamp_valid_bits = 0;
    float timestamp_period_ns = 0.0f;
    bool subgroup_supported = false;
    bool Vulkan11 = false;
    double last_gpu_ms = -1.0;
    GpuBuffer weights;
    GpuBuffer x;
    GpuBuffer y;
    std::string gpu_name;

    VulkanContext() = default;
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    ~VulkanContext() {
        if (device) {
            // No submit should still be active: the runner waits for each fence.
            vkDeviceWaitIdle(device);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (query_pool) vkDestroyQueryPool(device, query_pool, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (module) vkDestroyShaderModule(device, module, nullptr);
            if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
            // GpuBuffer members are destroyed after this destructor body.
            // Therefore explicitly release their allocations before device destruction.
            release(weights);
            release(x);
            release(y);
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    static void release(GpuBuffer& b) {
        if (b.mapped) {
            vkUnmapMemory(b.device, b.memory);
            b.mapped = nullptr;
        }
        if (b.buffer) {
            vkDestroyBuffer(b.device, b.buffer, nullptr);
            b.buffer = VK_NULL_HANDLE;
        }
        if (b.memory) {
            vkFreeMemory(b.device, b.memory, nullptr);
            b.memory = VK_NULL_HANDLE;
        }
        b.device = VK_NULL_HANDLE;
    }

    void initialize() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "G2 Turbo Engine";
        // Subgroup properties require core 1.1. Fall back to 1.0 if necessary.
        uint32_t instance_version = VK_API_VERSION_1_0;
        auto version_fn = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        if (version_fn) check(version_fn(&instance_version), "vkEnumerateInstanceVersion");
        Vulkan11 = VK_VERSION_MAJOR(instance_version) > 1 ||
                   VK_VERSION_MINOR(instance_version) >= 1;
        app.apiVersion = Vulkan11 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0;

        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        check(vkCreateInstance(&ci, nullptr, &instance), "vkCreateInstance");

        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr),
              "vkEnumeratePhysicalDevices(count)");
        if (!count) throw std::runtime_error("No Vulkan physical GPU");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
              "vkEnumeratePhysicalDevices(list)");

        bool found = false;
        for (auto gpu : devices) {
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, families.data());
            for (uint32_t i = 0; i < family_count; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    families[i].queueCount >= 1) {
                    physical = gpu;
                    queue_family = i;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (!found) throw std::runtime_error("No Vulkan compute queue");
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);
        gpu_name = props.deviceName;
        timestamp_period_ns = props.limits.timestampPeriod;
        uint32_t count_families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count_families, nullptr);
        std::vector<VkQueueFamilyProperties> families(count_families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count_families, families.data());
        timestamp_valid_bits = families.at(queue_family).timestampValidBits;

        // Query only when both the API and physical device support Vulkan 1.1.
        const bool physical_11 = VK_VERSION_MAJOR(props.apiVersion) > 1 ||
                                 VK_VERSION_MINOR(props.apiVersion) >= 1;
        if (Vulkan11 && physical_11) {
            VkPhysicalDeviceSubgroupProperties subgroup{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
            };
            VkPhysicalDeviceProperties2 properties2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
            };
            properties2.pNext = &subgroup;
            vkGetPhysicalDeviceProperties2(physical, &properties2);
            const uint32_t needed = VK_SUBGROUP_FEATURE_BASIC_BIT |
                                    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            subgroup_supported = (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                                 ((subgroup.supportedOperations & needed) == needed) &&
                                 subgroup.subgroupSize > 0 && subgroup.subgroupSize <= 64;
        }

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qc.queueFamilyIndex = queue_family;
        qc.queueCount = 1;
        qc.pQueuePriorities = &priority;
        VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dc.queueCreateInfoCount = 1;
        dc.pQueueCreateInfos = &qc;
        check(vkCreateDevice(physical, &dc, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queue_family, 0, &queue);
    }

    void prepare(const std::vector<uint32_t>& packed,
                 const std::vector<float>& activations, int rows, int cols,
                 const std::vector<uint32_t>& spirv) {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        weights.create(device, memory_properties,
                       packed.size() * sizeof(uint32_t));
        x.create(device, memory_properties,
                 activations.size() * sizeof(float));
        y.create(device, memory_properties,
                 size_t(rows) * sizeof(float));
        weights.write(packed.data(), packed.size() * sizeof(uint32_t));
        x.write(activations.data(), activations.size() * sizeof(float));
        std::memset(y.mapped, 0, size_t(rows) * sizeof(float));
        y.flush();

        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lc{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
        };
        lc.bindingCount = 3;
        lc.pBindings = bindings;
        check(vkCreateDescriptorSetLayout(device, &lc, nullptr, &layout),
              "vkCreateDescriptorSetLayout");

        struct Push { uint32_t rows; uint32_t cols; };
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset = 0;
        range.size = sizeof(Push);
        VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plc.setLayoutCount = 1;
        plc.pSetLayouts = &layout;
        plc.pushConstantRangeCount = 1;
        plc.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(device, &plc, nullptr, &pipeline_layout),
              "vkCreatePipelineLayout");

        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = spirv.size() * sizeof(uint32_t);
        sm.pCode = spirv.data();
        check(vkCreateShaderModule(device, &sm, nullptr, &module),
              "vkCreateShaderModule");
        VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
        };
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pc{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
        };
        pc.stage = stage;
        pc.layout = pipeline_layout;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pc,
                                       nullptr, &pipeline), "vkCreateComputePipelines");

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 3;
        VkDescriptorPoolCreateInfo dpc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpc.maxSets = 1;
        dpc.poolSizeCount = 1;
        dpc.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(device, &dpc, nullptr, &pool),
              "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo dai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
        };
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(device, &dai, &descriptor_set),
              "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo buffer_info[3]{};
        GpuBuffer* buffers[3] = {&weights, &x, &y};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            buffer_info[i].buffer = buffers[i]->buffer;
            buffer_info[i].offset = 0;
            buffer_info[i].range = buffers[i]->length;
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_info[i];
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &cp, nullptr, &command_pool),
              "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ca{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
        };
        ca.commandPool = command_pool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &ca, &command),
              "vkAllocateCommandBuffers");
        if (timestamp_valid_bits > 0 && timestamp_period_ns > 0.0f) {
            VkQueryPoolCreateInfo qc{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qc.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qc.queryCount = 2;
            if (vkCreateQueryPool(device, &qc, nullptr, &query_pool) != VK_SUCCESS) {
                query_pool = VK_NULL_HANDLE;  // Driver lacks timestamps: wall time still works.
            }
        }
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
        if (query_pool) {
            vkCmdResetQueryPool(command, query_pool, 0, 2);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                query_pool, 0);
        }
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &descriptor_set,
                                0, nullptr);
        Push push{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        // Exactly one workgroup per output row, 64 lanes per workgroup.
        vkCmdDispatch(command, static_cast<uint32_t>(rows), 1, 1);
        if (query_pool) {
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                query_pool, 1);
        }

        // Explicit shader-write -> host-read visibility before fence wait.
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = y.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                             1, &barrier, 0, nullptr);
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

        VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device, &fc, nullptr, &fence), "vkCreateFence");
    }

    double submit_and_wait() {
        check(vkResetFences(device, 1, &fence), "vkResetFences");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        auto start = Clock::now();
        check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
        // 10-second guard per GPU dispatch to avoid indefinite driver hangs.
        check(vkWaitForFences(device, 1, &fence, VK_TRUE,
                              10'000'000'000ULL), "vkWaitForFences");
        const double wall = millis(start);
        last_gpu_ms = -1.0;
        if (query_pool) {
            uint64_t ticks[2]{0, 0};
            const VkResult result = vkGetQueryPoolResults(
                device, query_pool, 0, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (result == VK_SUCCESS) {
                const uint64_t mask = timestamp_valid_bits >= 64 ?
                    UINT64_MAX : ((uint64_t(1) << timestamp_valid_bits) - 1);
                const uint64_t dt = (ticks[1] - ticks[0]) & mask;
                last_gpu_ms = double(dt) * double(timestamp_period_ns) / 1e6;
            }
        }
        return wall;
    }

    std::vector<float> read_output(int rows) {
        y.invalidate();
        std::vector<float> output(static_cast<size_t>(rows));
        std::memcpy(output.data(), y.mapped, size_t(rows) * sizeof(float));
        return output;
    }
};

}  // namespace


// Each variant is isolated in its own context so one bad GPU pipeline cannot
// prevent the other alternatives or CPU results from being reported.
struct VariantResult {
    std::string name;
    std::string json;
    bool success = false;
    double wall_ms = 1e99;
};

VariantResult run_variant(
        AAssetManager* assets,
        const std::string& name,
        const std::vector<uint32_t>& packed,
        const std::vector<float>& activations,
        const std::vector<float>& reference,
        int rows, int cols, int repetitions) {
    VariantResult result{name, "", false, 1e99};
    const auto setup_start = Clock::now();
    try {
        VulkanContext ctx;
        ctx.initialize();
        if (name == "subgroup" && !ctx.subgroup_supported)
            throw std::runtime_error("Vulkan subgroup arithmetic unavailable on compute");
        const auto shader = shader_spirv(
            assets, std::string("shaders/q4_0_gemv") +
                        (name == "subgroup" ? "_subgroup" :
                         name == "tiled32" ? "_32" : "") + ".spv");
        ctx.prepare(packed, activations, rows, cols, shader);
        const double setup_ms = millis(setup_start);

        const double cold_wall_ms = ctx.submit_and_wait();
        const double cold_gpu_ms = ctx.last_gpu_ms;
        const std::vector<float> got = ctx.read_output(rows);
        double mse = 0, reference_energy = 0, max_error = 0;
        bool finite = true;
        for (int i = 0; i < rows; ++i) {
            const double d = double(got[i]) - double(reference[i]);
            finite = finite && std::isfinite(d);
            mse += d * d;
            reference_energy += double(reference[i]) * double(reference[i]);
            max_error = std::max(max_error, std::abs(d));
        }
        const double nrmse = std::sqrt(mse / std::max(reference_energy, 1e-30));
        const bool accurate = finite && std::isfinite(nrmse) && nrmse < 0.001;
        if (!accurate) {
            std::ostringstream o;
            o << "{\"variant\":\"" << name << "\",\"status\":\"incorrect\","
              << "\"nrmse\":" << nrmse << ",\"max_abs_error\":" << max_error
              << ",\"pass\":false}";
            result.json = o.str();
            return result;
        }

        double wall_sum = 0, wall_min = 1e99, wall_max = 0;
        double gpu_sum = 0;
        int gpu_count = 0;
        for (int rep = 0; rep < repetitions; ++rep) {
            const double wall = ctx.submit_and_wait();
            wall_sum += wall;
            wall_min = std::min(wall_min, wall);
            wall_max = std::max(wall_max, wall);
            if (ctx.last_gpu_ms >= 0) {
                gpu_sum += ctx.last_gpu_ms;
                ++gpu_count;
            }
        }
        result.success = true;
        result.wall_ms = wall_sum / repetitions;
        std::ostringstream o;
        o << std::fixed << std::setprecision(6)
          << "{\"variant\":\"" << name << "\","
          << "\"status\":\"ok\","
          << "\"device\":\"" << json_escape(ctx.gpu_name) << "\","
          << "\"gpu_timestamp_available\":" << (gpu_count > 0 ? "true" : "false") << ","
          << "\"timestamp_valid_bits\":" << ctx.timestamp_valid_bits << ","
          << "\"timestamp_period_ns\":" << ctx.timestamp_period_ns << ","
          << "\"subgroup_supported\":" << (ctx.subgroup_supported ? "true" : "false") << ","
          << "\"setup_upload_pipeline_ms\":" << setup_ms << ","
          << "\"cold_wall_ms\":" << cold_wall_ms << ","
          << "\"cold_gpu_timestamp_ms\":" << cold_gpu_ms << ","
          << "\"hot_dispatch_wall_mean_ms\":" << result.wall_ms << ","
          << "\"hot_dispatch_wall_min_ms\":" << wall_min << ","
          << "\"hot_dispatch_wall_max_ms\":" << wall_max << ","
          << "\"hot_gpu_timestamp_mean_ms\":" << (gpu_count ? gpu_sum / gpu_count : -1.0) << ","
          << "\"normalized_rmse\":" << nrmse << ","
          << "\"max_absolute_error\":" << max_error << ","
          << "\"pass\":true}";
        result.json = o.str();
        return result;
    } catch (const std::exception& e) {
        result.json = "{\"variant\":\"" + name +
            "\",\"status\":\"error\",\"message\":\"" +
            json_escape(e.what()) + "\",\"pass\":false}";
        return result;
    }
}

std::string vulkan_benchmark_json(
    const std::vector<uint32_t>& packed_q40,
    const std::vector<float>& activations,
    const std::vector<float>& reference,
    int rows, int cols, int repetitions, AAssetManager* assets) {
    // Baseline v0.2, 32-thread tiled and subgroup alternatives.
    const char* names[] = {"baseline64", "tiled32", "subgroup"};
    std::vector<VariantResult> variants;
    for (const char* name : names) {
        variants.push_back(run_variant(assets, name, packed_q40,
                                       activations, reference, rows, cols, repetitions));
    }
    const VariantResult* best = nullptr;
    for (const auto& v : variants) {
        if (v.success && (!best || v.wall_ms < best->wall_ms))
            best = &v;
    }
    std::ostringstream o;
    o << "{\"status\":\"" << (best ? "ok" : "error") << "\","
      << "\"best_variant\":\"" << (best ? best->name : "none") << "\","
      << "\"variants\":[";
    for (size_t i = 0; i < variants.size(); ++i) {
        if (i) o << ",";
        o << variants[i].json;
    }
    o << "]}";
    return o.str();
}
} // namespace g2
