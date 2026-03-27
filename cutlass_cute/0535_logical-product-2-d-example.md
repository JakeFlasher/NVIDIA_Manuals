---
title: "Logical Product 2-D Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#logical-product-2-d-example"
---

### [Logical Product 2-D Example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#logical-product-2-d-example)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#logical-product-2-d-example "Permalink to this headline")

We can use the by-mode `tiler` strategies previously developed to write multidimensional products as well.

![product2d.png](images/_______-_______-_-_-________1.png)

The above image demonstates the use of a `tiler` to apply `logical_product` by-mode. Despite this **not being the recommended approach**, the result is a rank-2 layout consisting of 2x5 row-major block that is tiled across a 3x4 column-major arrangement.

The reason **this is not the recommended approach** is that the `tiler B` in the above expression is highly unintuitive. In fact, it requires perfect knowledge of the shape and strides of `A` in order to construct. We would like to express “Tile Layout `A` according to Layout `B`” in a way that makes `A` and `B` independent and is much more intuitive.
