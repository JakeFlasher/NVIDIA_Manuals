---
title: "Example 3 – Reshape a layout into a matrix"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#example-3-reshape-a-layout-into-a-matrix"
---

#### [Example 3 – Reshape a layout into a matrix](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example-3-reshape-a-layout-into-a-matrix)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example-3-reshape-a-layout-into-a-matrix "Permalink to this headline")

`(10,2):(16,4)  o  (5,4):(1,5)`

This describes interpreting the layout `(10,2):(16,4)`
as a 5x4 matrix in a column-major order.

1. ` = (10,2):(16,4) o (5:1,4:5)`. Layout `(5,4):(1,5)` as concatenation of sublayouts.
2. ` = ((10,2):(16,4) o 5:1, (10,2):(16,4) o 4:5)`. Left distributivity.
  - `(10,2):(16,4) o 5:1 => (5,1):(16,4)`. Mod out the shape `5`.
  - `(10,2):(16,4) o 4:5 => (2,2):(80,4)`. Div out the stride `5`.
3. ` = ((5,1):(16,4), (2,2):(80,4))`. Composed Layout as concatenation of sublayouts.
4. ` = (5:16, (2,2):(80,4))`. By-mode coalesce.
5. ` = (5,(2,2))):(16,(80,4))`. Final composed layout.

We get exactly this result with CuTe
if we use compile-time shapes and strides.
The following C++ code prints `(_5,(_2,_2)):(_16,(_80,_4))`.

```cpp
Layout a = make_layout(make_shape (Int<10>{}, Int<2>{}),
                       make_stride(Int<16>{}, Int<4>{}));
Layout b = make_layout(make_shape (Int< 5>{}, Int<4>{}),
                       make_stride(Int< 1>{}, Int<5>{}));
Layout c = composition(a, b);
print(c);
```

If we use dynamic integers, the following C++ code prints `((5,1),(2,2)):((16,4),(80,4))`.

```cpp
Layout a = make_layout(make_shape (10, 2),
                       make_stride(16, 4));
Layout b = make_layout(make_shape ( 5, 4),
                       make_stride( 1, 5));
Layout c = composition(a, b);
print(c);
```

The results may _look_ different but are the mathematically the same. The 1s in the shape don’t affect the layout as a mathematical function from 1-D coordinates to integers or as a function from 2-D coordinates to integers. In the dynamic case, CuTe can not coalesce the dynamic size-1 modes to “simplify” the layout due to the static rank and type of the tuples containing them.
