#pragma once

#include <iostream>
#include <sstream>

inline void LogError(const char* message) {
    std::cerr << message << std::endl;
}

inline void LogError(const char* file, int line, const char* message) {
    std::cerr << "file: " 
              << file << "("
              << line << ") "
              << message << std::endl;
}

template <class InputIterator>
std::string Join(InputIterator begin, InputIterator end, const std::string& delim) {
    std::ostringstream os;
    if (begin != end) {
        std::copy(begin, std::prev(end), std::ostream_iterator<typename std::iterator_traits<InputIterator>::value_type>(os, delim.c_str()));
        os << *std::prev(end);
    }
    return os.str();
}



//#define LogError(MESSAGE) PfLogError(__FILE__, __LINE__, MESSAGE)