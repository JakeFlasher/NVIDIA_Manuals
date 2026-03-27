---
title: "Example 2 – Reshape a layout into a matrix"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#example-2-reshape-a-layout-into-a-matrix"
---

#### [Example 2 – Reshape a layout into a matrix](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example-2-reshape-a-layout-into-a-matrix)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example-2-reshape-a-layout-into-a-matrix "Permalink to this headline")

`20:2  o  (5,4):(4,1)`. Composition formulation.

This describes interpreting the layout `20:2`
as a 5x4 matrix in a row-major order.

1. ` = 20:2 o (5:4,4:1)`. Layout `(5,4):(4,1)` as concatenation of sublayouts.
2. ` = (20:2 o 5:4, 20:2 o 4:1)`. Left distributivity.
  - `20:2 o 5:4  =>  5:8`. Trivial case.
  - `20:2 o 4:1  =>  4:2`. Trivial case.
3. ` = (5:8, 4:2)`. Composed Layout as concatenation of sublayouts.
4. ` = (5,4):(8,2)`. Final composed layout.
