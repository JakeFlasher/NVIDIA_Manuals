---
title: "Extensive Mixed-Precision Data Type Support"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/index.html#extensive-mixed-precision-data-type-support"
---

## [Extensive Mixed-Precision Data Type Support](https://docs.nvidia.com/cutlass/latest#extensive-mixed-precision-data-type-support)[](https://docs.nvidia.com/cutlass/latest/#extensive-mixed-precision-data-type-support "Permalink to this headline")

To support a broad range of applications, CUTLASS offers comprehensive support for mixed-precision computations via both its C++ templates and Python interfaces. Supported data types include:

- **Floating-point types**: FP64, FP32, TF32, FP16, BF16
- **Tensor Core-emulated FP32**
- **8-bit floating-point formats**: e5m2 and e4m3
- **Block scaled types**: NVIDIA NVFP4 and OCP standard MXFP4, MXFP6, MXFP8
- **Narrow integer types**: 4-bit and 8-bit signed/unsigned integers
- **Binary types**: 1-bit data types

The Python DSL extends this support by enabling experimentation with optimal data type combinations using a NumPy-style API.
