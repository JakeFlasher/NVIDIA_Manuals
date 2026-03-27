---
title: "5.4.11.3. Double Precision"
section: "5.4.11.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#double-precision"
---

### [5.4.11.3. Double Precision](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#double-precision)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#double-precision "Permalink to this headline")

Tensor Cores support double-precision floating point operations on devices with compute capability 8.0 and higher. To use this new functionality, a `fragment` with the `double` type must be used. The `mma_sync` operation will be performed with the .rn (rounds to nearest even) rounding modifier.
