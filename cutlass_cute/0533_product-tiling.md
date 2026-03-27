---
title: "Product (Tiling)"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#product-tiling"
---

## [Product (Tiling)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#product-tiling)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#product-tiling "Permalink to this headline")

Finally, we can define the product of a Layout by another Layout. In this section, we’ll define `logical_product(Layout, Layout)`, which again considers all `Layout`s as 1-D functions from integers to integers, and then use that definition to create multidimensional `Layout` products.

Informally, `logical_product(A, B)` results in a two mode layout where the first mode is the layout `A` and the second mode is the layout `B` but with each element replaced by a “unique replication” of layout `A`.

Formally, this can be written as

$A \otimes B := (A, A^* \circ B)$

and implemented in CuTe as

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_product(Layout<LShape,LStride> const& layout,
                     Layout<TShape,TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout, size(layout)*cosize(tiler)), tiler));
}
```

Note that this is defined only in terms of concatenation, composition, and complement.

So what is that?

> where the first mode is the layout `A`

This is clearly just a copy of `A`.

> the second mode is the layout `B` but with each element replaced by a “unique replication” of layout `A`.

The “unique replication” of layout `A` sounds like complement, `A*`, up to the cosize of `B`. As we’ve seen in the `complement` section, this can be described as the “layout of the repetition of `A`”. If `A` is the “tile”, then `A*` is the layout of repetitions that are available for `B`.
