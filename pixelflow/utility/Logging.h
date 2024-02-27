#pragma once

#include <iostream>


inline void LogError(const char* file, int line, const char* message) {
    std::cerr << "file: " 
              << file << "("
              << line << ") "
              << message << std::endl;
}


//#define LogError(MESSAGE) PfLogError(__FILE__, __LINE__, MESSAGE)