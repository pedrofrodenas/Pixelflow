#include "pixelflow/utility/StringUtils.h"

namespace pixelflow {
namespace utility {

std::vector<std::string> SplitString(const std::string& str,
                                     const std::string& delimiters /* = " "*/,
                                     bool trim_empty_str /* = true*/) {
    std::vector<std::string> tokens;
    std::string::size_type pos = 0, new_pos = 0, last_pos = 0;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
    return tokens;
}

std::string ToUpper(const std::string& input) {
    std::string output = input;
    std::transform(input.begin(), input.end(), output.begin(),
                    [](unsigned char c) { return std::toupper(c);});
    return output;
}

std::string ToLower(const std::string& input) {
    std::string output = input;
    std::transform(input.begin(), input.end(), output.begin(),
                    [](unsigned char c) { return std::tolower(c);});
    return output;
}

} // namespace utility
} // namespace pixelflow

