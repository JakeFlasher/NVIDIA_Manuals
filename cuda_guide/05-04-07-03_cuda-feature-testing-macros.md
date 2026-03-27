---
title: "5.4.7.3. CUDA Feature Testing Macros"
section: "5.4.7.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#cuda-feature-testing-macros"
---

### [5.4.7.3. CUDA Feature Testing Macros](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-feature-testing-macros)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-feature-testing-macros "Permalink to this headline")

`nvcc` provides the following preprocessor macros for feature testing. The macros are defined when a particular feature is supported by the CUDA front-end compiler.

- `__CUDACC_DEVICE_ATOMIC_BUILTINS__`: Supports [device atomic compiler builtins](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-atomic-functions).
- `__NVCC_DIAG_PRAGMA_SUPPORT__`: Supports [diagnostic control pragmas](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-diagnostic-pragmas).
- `__CUDACC_EXTENDED_LAMBDA__`: Supports [extended lambdas](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambdas). Enabled by  `--expt-extended-lambda` or `--extended-lambda` flag.
- `__CUDACC_RELAXED_CONSTEXPR__`: Support for [relaxed constexpr functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#constexpr-functions). Enabled by the `--expt-relaxed-constexpr` flag.
