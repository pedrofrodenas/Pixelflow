#include "pixelflow/core/dtypes.h"


namespace pixelflow {
namespace core {

static_assert(sizeof(int     ) == 4, "Unsupported platform: int must be 4 bytes."     );
static_assert(sizeof(int64_t ) == 8, "Unsupported platform: int64_t must be 8 bytes." );
static_assert(sizeof(int32_t ) == 4, "Unsupported platform: int32_t must be 4 bytes." );
static_assert(sizeof(int8_t  ) == 1, "Unsupported platform: int64_t must be 1 bytes." );
static_assert(sizeof(float    ) == 4, "Unsupported platform: float must be 4 bytes." );
static_assert(sizeof(double  ) == 8, "Unsupported platform: double must be 8 bytes."  );
static_assert(sizeof(bool    ) == 1, "Unsupported platform: bool must be 1 byte."     );

const PfType PfType::Undefined (PfType::PfTypeCode::Undefined,  1, "Undefined");
const PfType PfType::Float32  (PfType::PfTypeCode::Float,     4, "Float32"  );
const PfType PfType::Float64  (PfType::PfTypeCode::Float,     8, "Float64"  );
const PfType PfType::Int8     (PfType::PfTypeCode::Int,       1, "Int8"     );
const PfType PfType::Int32    (PfType::PfTypeCode::Int,       4, "Int32"    );
const PfType PfType::Int64    (PfType::PfTypeCode::Int,       8, "Int64"    );
const PfType PfType::UInt8    (PfType::PfTypeCode::UInt,      1, "UInt8"    );
const PfType PfType::Bool     (PfType::PfTypeCode::Bool,      1, "Bool"     );

const PfType Undefined = PfType::Undefined;
const PfType Float32 = PfType::Float32;
const PfType Float64 = PfType::Float64;
const PfType Int8 = PfType::Int8;
const PfType Int64 = PfType::Int64;
const PfType Int32 = PfType::Int32;
const PfType UInt8 = PfType::UInt8;
const PfType Bool = PfType::Bool;

PfType::PfType(PfTypeCode dtype_code, int64_t byte_size, const std::string &dataname)
    : dtype_code(dtype_code), byte_size(byte_size) {
    if (dataname.size() > max_name_len - 1) {
        std::ostringstream oss;
        oss << " Name "<< dataname << " must be shorter.";
        LogError(oss.str().c_str());
    } else {
        std::strncpy(name, dataname.c_str(), max_name_len);
        name[max_name_len - 1] = '\0';
    }
}

bool PfType::operator==(const PfType &other) const {
    bool rt = true;
    rt = rt && (dtype_code == other.dtype_code);
    rt = rt && (byte_size == other.byte_size);
    rt = rt && (std::strcmp(name, other.name) == 0);
    return rt;
}


} // namespace core
} // namespace pixelflow