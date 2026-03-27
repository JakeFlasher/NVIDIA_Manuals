---
title: "sgemm_2.cu"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#sgemm-2-cu"
---

## [sgemm_2.cu](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#sgemm-2-cu)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#sgemm-2-cu "Permalink to this headline")

An example that uses more complex `TiledMMA` and `TiledCopy` to perform partitioning in place of the `tA`, `tB`, and `tC` thread layouts. With this example, we try to emphasize that the shared memory layouts, the partitioning patterns, and the PTX instruction to use in each stage can be specified independently.
