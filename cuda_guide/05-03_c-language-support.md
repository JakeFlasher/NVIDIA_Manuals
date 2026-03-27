---
title: "5.3. C++ Language Support"
section: "5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#c-language-support"
---

# [5.3. C++ Language Support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#c-language-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#c-language-support "Permalink to this headline")

`nvcc` processes CUDA and device code according to the following specifications:

- **C++03** (ISO/IEC 14882:2003), `--std=c++03` flag.
- **C++11** (ISO/IEC 14882:2011), `--std=c++11` flag.
- **C++14** (ISO/IEC 14882:2014), `--std=c++14` flag.
- **C++17** (ISO/IEC 14882:2017), `--std=c++17` flag.
- **C++20** (ISO/IEC 14882:2020), `--std=c++20` flag.

Passing `nvcc` `-std=c++<version>` flag turns on all C++ features related to the specified version and also invokes the host preprocessor, compiler and linker with the corresponding C++ dialect option.

The compiler supports all language features of the supported standards, subject to the restrictions reported in the following sections.
