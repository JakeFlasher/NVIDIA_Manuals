---
title: "Convolution CMake Examples"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#convolution-cmake-examples"
---

## [Convolution CMake Examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#convolution-cmake-examples)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#convolution-cmake-examples "Permalink to this headline")

**Example.** All convolution kernels targeting NVIDIA Ampere’s 16816 Tensor Core operation

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='80' -DCUTLASS_LIBRARY_KERNELS=s16816fprop,s16816dgrad,s16816wgrad
```

**Example.** All forward propagation (fprop) convolution kernels targeting CUDA Cores for multiple NVIDIA architectures

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='50;60;61;70;75;80' -DCUTLASS_LIBRARY_KERNELS=sfprop
```

**Example.** All forward propagation (fprop) convolution kernels with FP32 accumulation and FP16 input targeting NVIDIA Ampere’s 16816 Tensor Core operation

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='80' -DCUTLASS_LIBRARY_KERNELS=s16816fprop_*_f16
```

**Example.** All backward weight gradient (wgrad) convolution kernels with FP32 accumulation, FP16 input, and optimized global memory iterator
targeting NVIDIA Ampere, Turing, and Volta Tensor Core operations

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='70;75;80' -DCUTLASS_LIBRARY_KERNELS=tensorop*s*wgrad_optimized_f16
```
