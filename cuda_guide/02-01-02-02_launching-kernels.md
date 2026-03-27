---
title: "2.1.2.2. Launching Kernels"
section: "2.1.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#launching-kernels"
---

### [2.1.2.2. Launching Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#launching-kernels)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#launching-kernels "Permalink to this headline")

The number of threads that will execute the kernel in parallel is specified as part of the kernel launch. This is called the execution configuration. Different invocations of the same kernel may use different execution configurations, such as a different number of threads or thread blocks.

There are two ways of launching kernels from CPU code, [triple chevron notation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-cpp-launching-kernels-triple-chevron) and `cudaLaunchKernelEx`. Triple chevron notation, the most common way of launching kernels, is introduced here. An example of launching a kernel using `cudaLaunchKernelEx` is shown and discussed in detail in in section [Section 3.1.1](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#advanced-host-cudalaunchkernelex).
