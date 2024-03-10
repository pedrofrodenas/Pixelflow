#include "pixelflow/core/Device.h"

namespace pixelflow {
namespace core {

static Device::DeviceType StringToDeviceType(const std::string& type_colon_id) {
    std::vector<std::string> elem = utility::SplitString(type_colon_id, ":");
    if (elem.size() == 2)
    {
        std::string deviceTypeLower = utility::ToLower(elem[0]);
        if ( deviceTypeLower == "cpu")
        {
            return Device::DeviceType::CPU;
        }
        else if ( deviceTypeLower == "cuda")
        {
            return Device::DeviceType::CUDA;
        }
        else
        {
            LogError("Incorrect Device Name");
        }
    }
    else
    {
        std::ostringstream oss;
        oss << "Invalid Device Type: " <<  type_colon_id << 
        "Valid device strings are like \"CPU:0\" or \"CUDA:1\"";
        LogError(oss.str().c_str());
    }
}

static int StringToDeviceId(const std::string& type_colon_id) {
    std::vector<std::string> elem = utility::SplitString(type_colon_id, ":");
    if (elem.size() == 2)
    {
        return std::stoi(elem[1]);
    }
    else
    {
        std::ostringstream oss;
        oss << "Invalid Device Type: " <<  type_colon_id << 
        "Valid device strings are like \"CPU:0\" or \"CUDA:1\"";
        LogError(oss.str().c_str());
    }
}

Device::Device(DeviceType device_type, int device_id)
    : device_type_(device_type), device_id_(device_id) {}

Device::Device(std::string device_type, int device_id)
    : Device(device_type + ":" + std::to_string(device_id)) {}

Device::Device(const std::string& device_and_id)
    : Device(StringToDeviceType(device_and_id),
             StringToDeviceId(device_and_id)) {}

std::vector<Device> Device::GetAvailableDevices() {
    const std::vector<Device> cpu_devices = Device::GetAvailableCPUDevices();
    const std::vector<Device> gpu_devices = Device::GetAvailableCUDADevices();

    std::vector<Device> devices;

    devices.insert(devices.end(), cpu_devices.begin(), cpu_devices.end());
    devices.insert(devices.end(), gpu_devices.begin(), gpu_devices.end());
    return devices;
}

std::vector<Device> Device::GetAvailableCPUDevices() {
    return {Device(Device::DeviceType::CPU, 0)};
}

std::vector<Device> Device::GetAvailableCUDADevices() {

    std::vector<Device> devicesFound;
    for (size_t i = 0; i != cuda::DeviceCount(); ++i) {
        devicesFound.push_back(Device(Device::DeviceType::CUDA, i));
    }
    return devicesFound;
}

std::string Device::ToString() const {
    std::string device = "";
    switch (device_type_)
    {
    case DeviceType::CPU:
        device += "CPU";
        break;
    case DeviceType::CUDA:
        device += "CUDA";
        break;
    default:
        LogError("Not supported device type");
        break;
    }
    device += ":" + std::to_string(device_id_);
    return device;
}

void Device::PrintAvailableDevices() {
    std::vector<Device> devices = GetAvailableDevices();
    for (const Device &d : devices) {
        std::cout << "Device: " << d.ToString();
    }
}

} // namespace core
} // namespace pixelflow