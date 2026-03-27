---
title: "Blocked and Raked Products"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#blocked-and-raked-products"
---

#### [Blocked and Raked Products](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#blocked-and-raked-products)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#blocked-and-raked-products "Permalink to this headline")

The `blocked_product(LayoutA, LayoutB)` and `raked_product(LayoutA, LayoutB)` are rank-sensitive transformations on top of 1-D `logical_product` that let us express the more intuitive `Layout` products that we most often want to express.

A key observation in the implementation of these functions are the compatibility post-conditions of `logical_product`:

```console
// @post rank(result) == 2
// @post compatible(layout_a, layout<0>(result))
// @post compatible(layout_b, layout<1>(result))
```

Because `A` is always compatible with mode-0 of the result and `B` is always compatible with mode-1 of the result, if we made `A` and `B` the same rank then we could “reassociate” like-modes after the product. That is, the “column” mode in `A` could be combined with the “column” mode in `B` and the “row” mode in `A` could be combined with the “row” mode in `B`, etc.

This is exactly what `blocked_product` and `raked_product` do and it is why they are called rank-sensitive. Unlike other CuTe functions that take `Layout` arguments, these care about the top-level rank of the arguments so that each mode can be reassociated after the `logical_product`.

![productblocked2d.png](images/_______-___-_____-_________1.png)

The above image shows the same result as the `tiler` approach, but with much more intuitive arguments. A 2x5 row-major layout is arranged as a tile in a 3x4 column-major arrangement. Also note that `blocked_product` went ahead and `coalesced` mode-0 for us.

Similarly, `raked_product` combines the modes slightly differently. Instead of the resulting “column” mode being constructed from the `A` “column” mode then the `B` “column” mode, the resulting “column” mode is constructed from the `B` “column” mode then the `A` “column” mode.

![productraked2d.png](images/_______-___-_____-_________2.png)

This results in the “tile” `A` now being interleaved or “raked” with the “layout-of-tiles” `B` instead of appearing as blocks. Other references call this a “cyclic distribution.”
