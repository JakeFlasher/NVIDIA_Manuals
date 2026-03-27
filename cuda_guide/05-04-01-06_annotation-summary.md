---
title: "5.4.1.6. Annotation Summary"
section: "5.4.1.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#annotation-summary"
---

### [5.4.1.6. Annotation Summary](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#annotation-summary)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#annotation-summary "Permalink to this headline")

The following table summarizes the CUDA annotations and reports which execution space each annotation applies to and where it is valid.

| Annotation | `__host__` / `__device__` / `__host__  __device__` | `__global__` |
| --- | --- | --- |
| [__noinline__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-specifiers), [__forceinline__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-specifiers), [__inline_hint__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-specifiers) | Function | ❌ |
| [__restrict__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#restrict) | Pointer Parameter | Pointer Parameter |
| [__grid_constant__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#grid-constant) | ❌ | Parameter |
| [__launch_bounds__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#launch-bounds) | ❌ | Function |
| [__maxnreg__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#maximum-number-of-registers-per-thread) | ❌ | Function |
| [__cluster_dims__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cluster-dimensions) | ❌ | Function |
