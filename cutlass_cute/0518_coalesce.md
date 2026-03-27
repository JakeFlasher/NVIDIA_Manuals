---
title: "Coalesce"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#coalesce"
---

## [Coalesce](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#coalesce)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#coalesce "Permalink to this headline")

In the previous section, we summarized `Layout`s with

> Layouts are functions from integers to integers.

The `coalesce` operation is a “simplify” on functions from integers to integers. If we only care about input integers, then we can manipulate the shape and number of modes of the `Layout` without changing it as a function. The only thing `coalesce` can’t change is the `Layout`’s `size`.

More specifically, you can find the checked post-conditions in [the `coalesce` unit test](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/coalesce.cpp), which we’ll reproduce here:

```cpp
// @post size(@a result) == size(@a layout)
// @post depth(@a result) <= 1
// @post for all i, 0 <= i < size(@a layout), @a result(i) == @a layout(i)
Layout coalesce(Layout const& layout)
```

For example,

```cpp
auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(layout);    // _12:_1
```

where we can see the result has fewer modes and is “simpler.” Indeed, this could save us a few operations in the coordinate mapping and index mapping (if those are performed dynamically).

So, how do we get there?

- We’ve already seen that column-major `Layout`s like `(_2,_4):(_1,_2)` act identically to `_8:_1` for 1-D coordinates.
- Modes with size static-1 will always produce a natural coordinate of static-0. They can be ignored no matter the stride.

Generalizing, consider a layout with just two integral modes, s0:d0 and s1:d1.  Denote the result of coalescing this layout as s0:d0 ++ s1:d1. Then, there are four cases:

1. `s0:d0  ++  _1:d1  =>  s0:d0`. Ignore modes with size static-1.
2. `_1:d0  ++  s1:d1  =>  s1:d1`. Ignore modes with size static-1.
3. `s0:d0  ++  s1:s0*d0  =>  s0*s1:d0`. If the second mode’s stride is the product of the first mode’s size and stride, then they can be combined.
4. `s0:d0  ++  s1:d1  =>  (s0,s1):(d0,d1)`. Else, nothing can be done and they must be treated separately.

That’s it! We can flatten any layout and apply the above binary operation to each pair of adjacent modes in order to “coalesce” the modes of the layout.
