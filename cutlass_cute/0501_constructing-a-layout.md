---
title: "Constructing a Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#constructing-a-layout"
---

### [Constructing a Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#constructing-a-layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#constructing-a-layout "Permalink to this headline")

A `Layout` can be constructed in many different ways.
It can include any combination of compile-time (static) integers
or run-time (dynamic) integers.

```c++
Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);

Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                             make_stride(Int<12>{},Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                               LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

The `make_layout` function returns a `Layout`.
It deduces the types of the function’s arguments and returns a `Layout` with the appropriate template arguments.
Similarly, the `make_shape` and `make_stride` functions
return a `Shape` resp. `Stride`.
CuTe often uses these `make_*` functions
due to restrictions around constructor template argument deduction (CTAD) and to avoid having to repeat static or dynamic integer types.

When the `Stride` argument is omitted, it is generated from the provided `Shape` with `LayoutLeft` as default. The `LayoutLeft` tag constructs strides as an exclusive prefix product of the `Shape` from left to right, without regard to the `Shape`’s hierarchy. This can be considered a “generalized column-major stride generation”. The `LayoutRight` tag constructs strides as an exclusive prefix product of the `Shape` from right to left, without regard to the `Shape`’s hierarchy. For shapes of depth one, this can be considered a “row-major stride generation”, but for hierarchical shapes the resulting strides may be surprising. For example, the strides of `s2xh4` above could be generated with `LayoutRight`.

Calling `print` on each layout above results in the following

```console
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

The `Shape:Stride` notation is used quite often for `Layout`. The `_N` notation is shorthand for a static integer while other integers are dynamic integers. Observe that both `Shape` and `Stride` may be composed of both static and dynamic integers.

Also note that the `Shape` and `Stride` are assumed to be _congruent_. That is, `Shape` and `Stride` have the same tuple profiles. For every integer in `Shape`, there is a corresponding integer in `Stride`. This can be asserted with

```cpp
static_assert(congruent(my_shape, my_stride));
```
