---
title: "5.4.2.1. Host Compiler Type Extensions"
section: "5.4.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#host-compiler-type-extensions"
---

### [5.4.2.1. Host Compiler Type Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#host-compiler-type-extensions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#host-compiler-type-extensions "Permalink to this headline")

The use of non-standard arithmetic types is permitted by CUDA, as long as the host compiler supports it. The following types are supported:

- 128-bit integer type `__int128`.
  - Supported on Linux when the host compiler defines the `__SIZEOF_INT128__` macro.
- 128-bit floating-point types `__float128` and `_Float128` are available on GPU devices with compute capability 10.0 and later. A constant expression of `__float128` type may be processed by the compiler in a floating-point representation with lower precision.
  - Supported on Linux x86 when the host compiler defines the `__SIZEOF_FLOAT128__` or `__FLOAT128__` macros.
- `_Complex` [types](https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Complex-Data-Types.html) are only supported in host code.
