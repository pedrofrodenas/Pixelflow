#pragma once

#include <cstdint>
#include <string>
#include <cstring>
#include <sstream>

#include "pixelflow/utility/Logging.h"

namespace pixelflow {
namespace core {

/* This kind of design is often used in systems that need to handle 
different data types in a generic way, such as in data processing
libraries or machine learning frameworks. It allows the system to
check the data type of a variable at runtime and handle it appropriately. 
It also provides a way to get the size of the data type and a human-readable
name for debugging or logging purposes.
*/

class PfType {
public:
    static const PfType Undefined;
    static const PfType Float32;
    static const PfType Float64;
    static const PfType Int64;
    static const PfType Int32;
    static const PfType Int8;
    static const PfType UInt8;

    enum class PfTypeCode {
        Undefined,
        Int,
        UInt,
        Float,
    };

    // Default constructor
    PfType() : PfType(PfTypeCode::Undefined, 1, "Undefined") {}

    explicit PfType(PfTypeCode dtype_code,
                   int64_t byte_size,
                   const std::string &name);

    std::string ToString() const { return name; }

    bool operator==(const PfType& other) const;

    bool operator!=(const PfType& other) const { return !(*this == other); }

    int64_t getSize() const { return byte_size; }
private:
    PfTypeCode dtype_code;
    std::int64_t byte_size;
    static constexpr size_t max_name_len = 12;
    char name[max_name_len];
};

extern const PfType Float32;

}
}