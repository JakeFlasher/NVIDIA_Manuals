---
title: "Use CUTLASS’s and CuTe’s fundamental types and operations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#use-cutlass-s-and-cute-s-fundamental-types-and-operations"
---

#### [Use CUTLASS’s and CuTe’s fundamental types and operations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#use-cutlass-s-and-cute-s-fundamental-types-and-operations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#use-cutlass-s-and-cute-s-fundamental-types-and-operations "Permalink to this headline")

Use the
[fundamental types and operations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html)
defined in CUTLASS consistently.
This contributes to a framework of interoperable, consistent components.
It reduces code duplication, which reduces build and test times.
It also saves developer effort.

CUTLASS’s fundamental types and operations include

- [Numeric types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#numeric-types) to represent numeric data in host and device code, and
- [functional.h](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#functional) to perform numeric operations in generic code.

CUTLASS 3.0 uses CuTe components to represent data layouts and multidimensional arrays.
Please refer to the [CuTe Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) for details.
CuTe has replaced CUTLASS 2.x components such as
[Containers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#containers),
[Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html), and
[`TensorRef` and `TensorView`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html#tensorref).
