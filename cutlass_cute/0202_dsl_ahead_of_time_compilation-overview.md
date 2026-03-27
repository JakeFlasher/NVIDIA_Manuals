---
title: "Overview"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html#dsl_ahead_of_time_compilation--overview"
---

## [Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#overview)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#overview "Permalink to this headline")

CuTe DSL Ahead-of-Time (hereinafter referred to as AOT) compilation allows you to:

- **Compile once, enable cross-compilation**: Write kernels in Python and cross-compile them for multiple GPU architectures.
- **Remove JIT overhead**: Eliminate compilation delays in production by pre-compiling kernels.
- **Flexible integration**: Easily integrate compiled kernels into both Python and C/C++ codebases using flexible deployment options.

We provide 2 levels of AOT ABI:

1. **Low-Level CuTe ABI**: This ABI is expressed using CuTe DSL types and tensors, mirroring the original Python function.
2. **High-Level Apache TVM FFI ABI**: For interop with various frameworks (e.g., PyTorch, JAX), and offer high-level stable ABI access.

This guide will focus on the CuTe ABI AOT. For the Apache TVM FFI AOT, please refer to the section “Exporting Compiled Module” in [Compile with TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html).
