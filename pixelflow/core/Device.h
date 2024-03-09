#pragma once

#include <string>

#include "pixelflow/utility/StringUtils.h"
#include "pixelflow/utility/Logging.h"

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

    /// Constructor from string, e.g. "CUDA:0".
    explicit Device(const std::string& device_and_id);

    // Returns true iff device type is CPU.
    inline bool IsCPU() const { return device_type_ == DeviceType::CPU; }

    // Returns true iff device type is CUDA.
    inline bool IsCUDA() const { return device_type_ == DeviceType::CUDA; }

protected:
    DeviceType device_type_ = DeviceType::CPU;
    int device_id_ = 0;   
};

} // namespace core
} // namespace pixelflow