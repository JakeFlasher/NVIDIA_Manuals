---
title: "Beyond GEMM: Convolution Support"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/index.html#beyond-gemm-convolution-support"
---

## [Beyond GEMM: Convolution Support](https://docs.nvidia.com/cutlass/latest#beyond-gemm-convolution-support)[](https://docs.nvidia.com/cutlass/latest/#beyond-gemm-convolution-support "Permalink to this headline")

Beyond GEMM, CUTLASS supports high-performance convolution operations through the **implicit GEMM algorithm**. Implicit GEMM reformulates convolution operations as matrix multiplications (GEMM), enabling CUTLASS to leverage its modular and highly optimized GEMM pipeline. This approach allows CUTLASS to construct efficient convolutions by reusing highly optimized GEMM components.
