#pragma once

#include <vector>
#include <string>

namespace pixelflow {
namespace utility {

// http://stackoverflow.com/questions/236129/split-a-string-in-c
std::vector<std::string> SplitString(const std::string& str,
                                     const std::string& delimiters = " ",
                                     bool trim_empty_str = true);

std::string ToUpper(const std::string& input);

std::string ToLower(const std::string& input);

} // namespace utility
} // namespace pixelflow