---
title: "IntTuple"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#inttuple"
---

### [IntTuple](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#inttuple)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#inttuple "Permalink to this headline")

CuTe defines the IntTuple concept as either an integer, or a tuple of IntTuples. Note the recursive definition.
In C++, we define [operations on `IntTuple`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/int_tuple.hpp).

Examples of `IntTuple`s include:

- `int{2}`, the dynamic integer 2.
- `Int<3>{}`, the static integer 3.
- `make_tuple(int{2}, Int<3>{})`, the tuple of dynamic-2, and static-3.
- `make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})`, the tuple of dynamic-42, tuple of static-1 and dynamic-3, and static-17.

CuTe reuses the `IntTuple` concept for many different things,
including Shape, Stride, Step, and Coord
(see [`include/cute/layout.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp)).

Operations defined on `IntTuple`s include the following.

- `rank(IntTuple)`: The number of elements in an `IntTuple`. A single integer has rank 1, and a tuple has rank `tuple_size`.
- `get<I>(IntTuple)`: The `I`th element of the `IntTuple`, with `I < rank`. For single integers, `get<0>` is just that integer.
- `depth(IntTuple)`: The number of hierarchical `IntTuple`s. A single integer has depth 0, a tuple of integers has depth 1, a tuple that contains a tuple of integers has depth 2, etc.
- `size(IntTuple)`: The product of all elements of the `IntTuple`.

We write `IntTuple`s with parentheses to denote the hierarchy. For example, `6`, `(2)`, `(4,3)`, and `(3,(6,2),8)` are all `IntTuple`s.
