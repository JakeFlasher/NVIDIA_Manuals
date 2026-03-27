---
title: "5.4.11. Warp Matrix Functions"
section: "5.4.11"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions"
---

## [5.4.11. Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-matrix-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-matrix-functions "Permalink to this headline")

C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form `D=A*B+C`. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a [warp](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-warps-simt). In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire [warp](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-warps-simt), otherwise the code execution is likely to hang.
