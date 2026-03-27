---
title: "CUDA Built-in Variables"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#cuda-built-in-variables"
---

#### [CUDA Built-in Variables](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cuda-built-in-variables)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cuda-built-in-variables "Permalink to this headline")

Avoid direct access to CUDA built-in variables `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` within
CUTLASS components except in special circumstances.

Using built-in global variables directly within resuable components necessitates that all components
use them consistently which may not be possible if CUTLASS components are used in other contexts.

Instead, components should accept a linear ID identifying threads, warps, and threadblocks from calling
code. The top-level kernel may then decide how to map threads, warps, and blocks to the problem it is
solving.
