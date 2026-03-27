---
title: "3.1.1. cudaLaunchKernelEx"
section: "3.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#cudalaunchkernelex"
---

## [3.1.1. cudaLaunchKernelEx](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#cudalaunchkernelex)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#cudalaunchkernelex "Permalink to this headline")

When the [triple chevron notation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-launching-kernels-triple-chevron) was introduced in first versions of, the [Kernel Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-configuration) of a kernel had only four programmable parameters:
- thread block dimensions
- grid dimensions
- dynamic shared-memory (optional, 0 if unspecified)
- stream (default stream used if unspecified)

Some CUDA features can benefit from additional attributes and hints provided with a kernel launch. The `cudaLaunchKernelEx` enables a program to set the above mentioned execution configuration parameters via the `cudaLaunchConfig_t` structure.  In addition, the `cudaLaunchConfig_t` structure allows the program to pass in zero or more `cudaLaunchAttributes` to control or suggest other parameters for the kernel launch. For example, the `cudaLaunchAttributePreferredSharedMemoryCarveout` discussed later in this chapter (see [Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernel-l1-shared-config)) is specified using `cudaLaunchKernelEx`. The `cudaLaunchAttributeClusterDimension` attribute, discussed later in this chapter, is used to specify the desired cluster size for the kernel launch.

The complete list of supported attributes and their meaning is captured in the [CUDA Runtime API Reference Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gfc5ed48085f05863b1aeebb14934b056).
