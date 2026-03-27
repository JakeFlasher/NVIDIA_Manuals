---
title: "Overview"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#dsl_introduction--overview"
---

## [Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#overview)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#overview "Permalink to this headline")

CuTe DSL is a Python-based domain-specific language (DSL) designed for dynamic compilation of
high-performance GPU kernels.  It evolved from the C++ CUTLASS library and is
now available as a decorator-based DSL.

Its primary goals are:

- **Zero-cost abstraction**, DSL is a zero-cost abstraction thanks to Hybrid DSL approach.
- **Consistent with CuTe C++**, allowing users to express GPU kernels with full
control of the hardware.
- **JIT compilation** for both host and GPU execution.
- [DLPack](https://github.com/dmlc/dlpack) **integration**, enabling seamless
interop with frameworks (e.g., PyTorch, JAX).
- **JIT caching**, so that repeated calls to the same function benefit from
cached IR modules.
- **Native types and type inference** to reduce boilerplate and improve
performance.
- **Optional lower-level control**, offering direct access to GPU backends or
specialized IR dialects.
