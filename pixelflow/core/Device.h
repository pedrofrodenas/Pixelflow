#pragma once

#include <string>

#include "pixelflow/utility/StringUtils.h"
#include "pixelflow/utility/Logging.h"
#include "pixelflow/core/cudaUtils.h"

namespace pixelflow {
namespace core {

class Device {

public:
    // Type for device.
    enum class DeviceType {
        CPU = 0,
        CUDA = 1,
    };

    // Default constructor -> "CPU:0".
    Device() = default;

    // Constructor with device specified.
    explicit Device(DeviceType device_type, int device_id);

    explicit Device(std::string device_type, int device_id);

    // Constructor from string, e.g. "CUDA:0".
    explicit Device(const std::string& device_and_id);

    // Returns type of the device, e.g. DeviceType::CPU, DeviceType::CUDA.
    inline DeviceType GetType() const { return device_type_; }

    // Returns the device index (within the same device type).
    inline int GetID() const { return device_id_; }

    // Returns true iff device type is CPU.
    inline bool IsCPU() const { return device_type_ == DeviceType::CPU; }

    // Returns true iff device type is CUDA.
    inline bool IsCUDA() const { return device_type_ == DeviceType::CUDA; }

    // Returns a vector of available devices.
    static std::vector<Device> GetAvailableDevices();

    static std::vector<Device> GetAvailableCPUDevices();
    // Return a vector of CUDA devices
    static std::vector<Device> GetAvailableCUDADevices();

protected:
    DeviceType device_type_ = DeviceType::CPU;
    int device_id_ = 0;   
};

} // namespace core
} // namespace pixelflow