---
title: "2.1.9.1. Detecting Device Compilation"
section: "2.1.9.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#detecting-device-compilation"
---

### [2.1.9.1. Detecting Device Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#detecting-device-compilation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#detecting-device-compilation "Permalink to this headline")

When a function is specified with `__host__ __device__`, the compiler is instructed to generate both a GPU and a CPU code for this function. In such functions, it may be desirable to use the preprocessor to specify code only for the GPU or the CPU copy of the function. Checking whether `__CUDA_ARCH_` is defined is the most common way of doing this, as illustrated in the example below.
