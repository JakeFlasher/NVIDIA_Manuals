---
title: "Next steps"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#next-steps"
---

## [Next steps](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#next-steps)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#next-steps "Permalink to this headline")

All of the above examples assume that the CTA tile size divides the problem size so that global memory loads do no need to be predicated. The
[predication section of the tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html)
explains what to do if a matrix tiling
doesn’t perfectly divide the matrix.
