---
title: "5.3.6.1. clock() and clock64()"
section: "5.3.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#clock-and-clock64"
---

### [5.3.6.1. clock() and clock64()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#clock-and-clock64)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#clock-and-clock64 "Permalink to this headline")

```cuda
__host__ __device__ clock_t   clock();
__device__          long long clock64();
```

When executed in device code, it returns the value of a per-multiprocessor counter that increments every clock cycle. Sampling this counter at the beginning and end of a kernel, subtracting the two values, and recording the result for each thread provides an estimate of the number of clock cycles the device spends executing the thread. However, this value does not represent the actual number of clock cycles the device spends executing the thread’s instructions. The former number is greater than the latter because threads are time-sliced.

> **Hint**
>
> - The corresponding [CUDA C++ function](https://en.cppreference.com/w/cpp/chrono/c/clock.html) `cuda::std::clock()` is provided in the `<cuda/std/ctime>` header.
> - A portable [C++](https://en.cppreference.com/w/cpp/header/chrono) `<chrono>` implementation is also provided in the `<cuda/std/chrono>` [header](https://nvidia.github.io/cccl/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time)  for similar purposes.
