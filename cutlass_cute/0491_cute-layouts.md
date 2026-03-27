---
title: "CuTe Layouts"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#cute-layouts"
---

# [CuTe Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cute-layouts)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cute-layouts "Permalink to this headline")

This document describes `Layout`, CuTe’s core abstraction.
Fundamentally, a `Layout` maps from coordinate space(s)
to an index space.

`Layout`s present a common interface to multidimensional array access
that abstracts away the details of how the array’s elements are organized in memory.
This lets users write algorithms that access multidimensional arrays generically,
so that layouts can change, without users’ code needing to change. For example, a row-major MxN layout and a column-major MxN layout can be treated identically in software.

CuTe also provides an “algebra of `Layout`s.”
`Layout`s can be combined and manipulated
to construct more complicated layouts
and to tile layouts across other layouts.
This can help users do things like partition layouts of data over layouts of threads.
