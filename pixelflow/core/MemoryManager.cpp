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

