---
title: "Templates"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#templates"
---

### [Templates](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#templates)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#templates "Permalink to this headline")

CUDA C++ templates and modern generic programming techniques enable CUTLASS device code to span a large design space.

This design space includes:

- Mixed precision arithmetic and data storage
- Kernels specialized for layout and problem size
- Support for kernel fusion

Moreover, templates provided a structured approach to collecting compile-time constants such as tile dimensions. These
must be template arguments to target static array allocation and take advantage of loop unrolling, constant folding,
and function inlining.
