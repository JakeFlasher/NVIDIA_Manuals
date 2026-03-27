---
title: "CUTLASS Convolution Implementation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#cutlass-convolution-implementation"
---

# [CUTLASS Convolution Implementation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-convolution-implementation)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-convolution-implementation "Permalink to this headline")

To get the best performance, the following parameters are recommended.

- All tensors are 128-bit aligned NHWC tensors
- Channel count (C) is a multiple of 32 elements
- Filter count (K) is a multiple of 32 elements

This enables 128-bit vector memory acceses which lead to efficient CUDA kernels. Smaller alignment is supported even on tensor cores by setting AlignmentA and AlignmentB in `conv::kernel::DefaultConv2dFprop`, but the performance is lower than 128-bit aligned tensors.
