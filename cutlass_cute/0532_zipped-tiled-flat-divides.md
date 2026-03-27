---
title: "Zipped, Tiled, Flat Divides"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#zipped-tiled-flat-divides"
---

### [Zipped, Tiled, Flat Divides](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#zipped-tiled-flat-divides)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#zipped-tiled-flat-divides "Permalink to this headline")

It’s easy to see the tiles when they are highlighted in the images above, but working with them can still be awkward. How would you slice out the `3`rd tile or the `7`th tile or the `(1,2)`th tile so you could continue working on it?

Enter the convenience flavors of `logical_divide`. Suppose we have a `Layout` and a `Tiler` of some shape, then each operation will apply `logical_divide`, but potentially rearrange the modes into more convenient forms.

```text
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

For example, the `zipped_divide` function applies `logical_divide`, and then gathers the “subtiles” into a single mode and the “rest” into a single mode.

```cpp
// A: shape is (9,32)
auto layout_a = make_layout(make_shape (Int< 9>{}, make_shape (Int< 4>{}, Int<8>{})),
                            make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
// B: shape is (3,8)
auto tiler = make_tile(Layout<_3,_3>{},           // Apply     3:3     to mode-0
                       Layout<Shape <_2,_4>,      // Apply (2,4):(1,8) to mode-1
                              Stride<_1,_8>>{});

// ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))
auto ld = logical_divide(layout_a, tiler);
// ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
auto zd = zipped_divide(layout_a, tiler);
```

Then, the offset to the `3`rd tile is `zd(0,3)`. The offset to the `7`th tile is `zd(0,7)`. The offset to the `(1,2)`th tile is `zd(0,make_coord(1,2))`. The tile itself always has layout `layout<0>(zd)`. Indeed, it is always the case that

`layout<0>(zipped_divide(a, b)) == composition(a, b)`.

We note that `logical_divide` preserves the _semantics_ of the modes while permuting the elements within those modes – the `M`-mode of layout `A` is still the `M`-mode of the result and the `N`-mode of layout `A` is still the `N`-mode of the result.

This is not the case with `zipped_divide`. The mode-0 in the `zipped_divide` result is the `Tile` itself (of whatever rank the `Tiler` was) and mode-1 is the layout of those tiles. It doesn’t always make sense to plot these as 2-D layouts, because the `M`-mode is now more aptly the “tile-mode” and the `N`-mode is more aptly the “rest-mode”. Regardless, we still can plot the resulting layout as 2-D as shown below.

![divide3.png](images/______-_____-____-________1.png)

We’ve kept each tile as its color in the previous images for clarity. Clearly, iterating across tiles is now equivalent to iterating across a row of this layout and iterating over elements within a tile is equivalent to iterating down a column of this layout. As we’ll see in the `Tensor` section, this can be used to great effect in partitioning within or across tiles of data.
