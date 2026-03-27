---
title: "sgemm_1.cu"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#sgemm-1-cu"
---

## [sgemm_1.cu](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#sgemm-1-cu)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#sgemm-1-cu "Permalink to this headline")

The simplest of the tutorial examples covers the basics of partitioning the global memory into tiles across the CTAs (also called threadblocks in CUDA), partitioning the data tiles across the threads of each CTA, and writing a mainloop using `cute::copy` and `cute::gemm`.
