---
title: "5.3.5. CUDA C++ Standard Library"
section: "5.3.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cuda-c-standard-library"
---

## [5.3.5. CUDA C++ Standard Library](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-c-standard-library)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-c-standard-library "Permalink to this headline")

CUDA provides an implementation of the C++ Standard Library (STL), called [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html). The library presents the following benefits:

- The functionalities are available on both host and device.
- Compatible with all [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id59) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#id2) platforms supported by the CUDA Toolkit.
- Compatible with all [GPU architectures](https://developer.nvidia.com/cuda-gpus) supported by the last two major versions of the CUDA Toolkit.
- Compatible with all [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with the current and previous major versions.
- Provides C++17 backports of C++ Standard Library features available in recent standard versions, including C++20, C++23, and C++26.
- Supports extended data types, such as 128-bit integers (`__int128`), half-precision floats (`__half`), Bfloat16 (`__nv_bfloat16`), and quad-precision floats (`__float128`).
- Highly optimized for device code.

In addition, `libcu++` provides [extended features](https://nvidia.github.io/cccl/libcudacxx/extended_api.html) that are not available in the C++ Standard Library to improve productivity and application performance. Such features include  mathematical functions, memory operations, synchronization primitives, container extensions, high-level abstractions of CUDA intrinsics, C++ PTX wrappers, and more.

`libcu++` is available as part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), as well as part of the open-source [CCCL](https://nvidia.github.io/cccl/) repository.
