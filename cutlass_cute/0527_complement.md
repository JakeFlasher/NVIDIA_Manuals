---
title: "Complement"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#complement"
---

## [Complement](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#complement)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#complement "Permalink to this headline")

Before getting to “product” and “divide,” we need one more operation. We can think of `composition` as a layout `B` that is “selecting” certain coordinates from another layout `A`. But what about the coordinates that aren’t “selected”? To implement generic tiling, we want to be able to select arbitrary elements – the tile – and to describe the layout of those tiles – the leftovers, or the “rest.”

The `complement` of a layout attempts to find another layout that represents the “rest” – the elements that aren’t touched by the layout.

You can find many examples and checked post-conditions in [the `complement` unit test](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/complement.cpp). The post-conditions include

```cpp
// @post cosize(make_layout(@a layout_a, @a result))) >= size(@a cotarget)
// @post cosize(@a result) >= round_up(size(@a cotarget), cosize(@a layout_a))
// @post for all i, 1 <= i < size(@a result),
//         @a result(i-1) < @a result(i)
// @post for all i, 1 <= i < size(@a result),
//         for all j, 0 <= j < size(@a layout_a),
//           @a result(i) != @a layout_a(j)
Layout complement(LayoutA const& layout_a, Shape const& cotarget)
```

That is, the complement `R` of a layout `A` with respect to a Shape (IntTuple) `M` satisfies the following properties.

1. The size (and cosize) of `R` is _bounded_ by `size(M)`.
2. `R` is _ordered_.  That is, the strides of `R` are positive and increasing.  This means that `R` is unique.
3. `A` and `R` have _disjoint_ codomains. `R` attempts to “complete” the codomain of `A`.

The `cotarget` parameter above is most commonly an integer – you can see we only use `size(cotarget)` above. However, sometimes it is useful to specify an integer that has static properties. For example, `28` is a dynamic integer and `(_4,7)` is a shape with size `28` that is statically known to be divisible by `_4`. Both will produce the same `complement` mathematically, but the extra information can used by `complement` to preserve the staticness of the result as much as possible.
