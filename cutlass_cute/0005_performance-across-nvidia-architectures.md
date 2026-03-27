---
title: "Performance Across NVIDIA Architectures"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/index.html#performance-across-nvidia-architectures"
---

## [Performance Across NVIDIA Architectures](https://docs.nvidia.com/cutlass/latest#performance-across-nvidia-architectures)[](https://docs.nvidia.com/cutlass/latest/#performance-across-nvidia-architectures "Permalink to this headline")

CUTLASS delivers peak-performance matrix multiplication on NVIDIA Tensor Core architectures, from the **NVIDIA Ampere Architecture** through the **NVIDIA Blackwell Architecture**, using both optimized C++ templates and Python-generated kernels.

The Python DSL is optimized for the **NVIDIA Blackwell Architecture** and achieves performance within 2% of handwritten C++ implementations, while reducing development time through just-in-time (JIT) compilation and interactive debugging.

Python support for the **NVIDIA Hopper Architecture** and **NVIDIA Ampere Architecture** will be available as an experimental feature at launch, with improvements planned in future releases.
