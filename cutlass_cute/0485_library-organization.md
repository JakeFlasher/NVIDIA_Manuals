---
title: "Library Organization"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html#library-organization"
---

## [Library Organization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#library-organization)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#library-organization "Permalink to this headline")

CuTe is a header-only C++ library, so there is no source code that needs building. Library headers are contained within the top level [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) directory, with components of the library grouped by directories that represent their semantics.

| Directory | Contents |
| --- | --- |
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | Each header in the top level corresponds to one of the fundamental building blocks of CuTe, such as [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) and [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp). |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | Implementations of STL-like objects, such as tuple, array, and aligned array. |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | Fundamental numeric data types that include nonstandard floating-point types, nonstandard integer types, complex numbers, and integer sequence. |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | Implementations of utility algorithms such as copy, fill, and clear that automatically leverage architecture-specific features if available. |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | Wrappers for architecture-specific matrix-matrix multiply and copy instructions. |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | Meta-information for instructions in `arch` and utilities like partitioning and tiling. |
