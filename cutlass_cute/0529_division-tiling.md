---
title: "Division (Tiling)"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#division-tiling"
---

## [Division (Tiling)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#division-tiling)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#division-tiling "Permalink to this headline")

Finally, we can define the division of a `Layout` by another `Layout`. Functions that divide a layout into components are useful as a basis for tiling and partitioning layouts.

In this section, we’ll define `logical_divide(Layout, Layout)`, which again considers all `Layout`s as 1-D functions from integers to integers, and then use that definition to create multidimensional `Layout` divides.

Informally, `logical_divide(A, B)` splits a layout `A` into two modes – in the first mode are all elements pointed to by `B` and in the second mode are all elements not pointed to by `B`.

Formally, this can be written as

$A \oslash B := A \circ (B,B^*)$

and implemented as

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_divide(Layout<LShape,LStride> const& layout,
                    Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));
}
```

Note that this is defined only in terms of concatenation, composition, and complement.

So what is that?

> in the first mode are all elements pointed to by `B`

This is clearly composition, `A o B`.

> in the second mode are all elements not pointed to by `B`

The elements NOT pointed to by `B` sounds like a complement, `B*`, up to the size of `A`. As we’ve seen above in the `complement` section, this can be described as the “layout of the repetition of `B`.” If `B` is the “tiler”, then `B*` is the layout of the tiles.
