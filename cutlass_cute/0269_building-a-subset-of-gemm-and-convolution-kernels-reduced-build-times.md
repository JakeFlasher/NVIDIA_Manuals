---
title: "Building a subset of GEMM and Convolution kernels (reduced build times)"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/overview.html#building-a-subset-of-gemm-and-convolution-kernels-reduced-build-times"
---

## [Building a subset of GEMM and Convolution kernels (reduced build times)](https://docs.nvidia.com/cutlass/latest#building-a-subset-of-gemm-and-convolution-kernels-reduced-build-times)[](https://docs.nvidia.com/cutlass/latest/#building-a-subset-of-gemm-and-convolution-kernels-reduced-build-times "Permalink to this headline")

To compile strictly one kernel or a small set of kernels, a comma-delimited list of kernel names with
wildcard characters may be used to reduce the set of kernels. The following examples show building exactly one
or a subset of kernels for NVIDIA Ampere and Turing architecture:
