#include "pixelflow/core/MemoryManager.h"

namespace pixelflow {
namespace core {

void* MemoryManager::Malloc(size_t byte_size, const Device& device) {
    std::shared_ptr<MemoryManagerDevice> memory_manager = GetMemoryManagerDevice(device);
    void* ptr = memory_manager->Malloc(byte_size, device);
    return ptr;
}

void MemoryManager::Free(void* ptr, const Device& device) {
    std::shared_ptr<MemoryManagerDevice> memory_manager = GetMemoryManagerDevice(device);
    memory_manager->Free(ptr, device);
}

void MemoryManager::Memcpy(void* dst_ptr,
                           const Device& dst_device,
                           const void* src_ptr,
                           const Device& src_device,
                           size_t num_bytes) {
    if (num_bytes == 0) {
        return;
    }
    if ( (dst_ptr == nullptr) || (src_ptr == nullptr)) {
        LogError("src_ptr and dst_ptr cannot be nullptr.");
    }
    std::shared_ptr<MemoryManagerDevice> memory_manager;

    // CPU memory manager
    if (src_device.IsCPU() && dst_device.IsCPU()) {
        memory_manager = GetMemoryManagerDevice(src_device);
    }
    else if (src_device.IsCPU() && dst_device.IsCUDA()) {
        memory_manager = GetMemoryManagerDevice(dst_device);
    }
    else if (src_device.IsCUDA() && dst_device.IsCPU()) {
        memory_manager = GetMemoryManagerDevice(src_device);
    }
    else if (src_device.IsCUDA() && dst_device.IsCUDA()) {
        memory_manager = GetMemoryManagerDevice(src_device);
    }
    else {
        std::ostringstream oss;
        oss << "Unsupported device type from " << src_device.ToString() << " to " << dst_device.ToString();
        LogError(oss.str().c_str());
    }
    memory_manager->Memcpy(dst_ptr, dst_device, src_ptr, src_device, num_bytes);
    }

std::shared_ptr<MemoryManagerDevice> MemoryManager::GetMemoryManagerDevice(const Device& device){
    // We are going to define an unordered_map that holds as a key 
    // Device::DeviceType as a value shared_pointers to MemoryManagerDevice 
    // object that manages the allocation of memory of different devices, 
    // this function return the suitable manager depending of the input device.

    static std::unordered_map<Device::DeviceType,
                              std::shared_ptr<MemoryManagerDevice>,
                              utility::hash_enum_class>
            map_device_type_to_memory_manager = {
                {Device::DeviceType::CPU, 
                 std::make_shared<MemoryManagerCPU>()}};

    // Check if selected device is not in the available
    // map_device_type_to_memory_manager shared_pointers
    // memory managers
    if (map_device_type_to_memory_manager.find(device.GetType()) ==
        map_device_type_to_memory_manager.end()) {
            std::ostringstream oss;
            oss << "Unsupported device " << device.ToString();
            LogError(oss.str().c_str());
    }
    return map_device_type_to_memory_manager.at(device.GetType());
}

} //namespace core
} // namespace pixelflow

