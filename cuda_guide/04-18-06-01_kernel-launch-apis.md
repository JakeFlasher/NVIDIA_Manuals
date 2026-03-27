---
title: "4.18.6.1. Kernel Launch APIs"
section: "4.18.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#kernel-launch-apis"
---

### [4.18.6.1. Kernel Launch APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#kernel-launch-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#kernel-launch-apis "Permalink to this headline")

Device-side kernel launches can be implemented using the following two APIs accessible from PTX: `cudaLaunchDevice()` and `cudaGetParameterBuffer()`. `cudaLaunchDevice()` launches the specified kernel with the parameter buffer that is obtained by calling `cudaGetParameterBuffer()` and filled with the parameters to the launched kernel. The parameter buffer can be NULL, i.e., no need to invoke `cudaGetParameterBuffer()`, if the launched kernel does not take any parameters.
