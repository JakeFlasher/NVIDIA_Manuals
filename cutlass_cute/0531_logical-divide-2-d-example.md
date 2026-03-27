---
title: "Logical Divide 2-D Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#logical-divide-2-d-example"
---

### [Logical Divide 2-D Example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#logical-divide-2-d-example)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#logical-divide-2-d-example "Permalink to this headline")

Using the `Tiler` concept defined above, this immediately generalizes to multidimensional tiling. The below example simply applies `layout_divide` by-mode to the cols and rows of a 2-D layout using a `Tiler`.

Similar to the 2-D composition example above, consider a 2-D layout `A = (9,(4,8)):(59,(13,1))` and want to apply `3:3` down the columns (mode-0) and `(2,4):(1,8)` across the rows (mode-1). This means the tiler can be written as `B = <3:3, (2,4):(1,8)>`.

![divide2.png](images/_______-______-_-_-________1.png)

The above figure depicts `A` as a 2-D layout with the elements pointed to by `B` highlighted in gray. The layout `B` describes our “tile” of data, and there are twelve of those tiles in `A` shown by each of the colors. After the divide, the first mode of each mode of the result is the tile of data and the second mode of each mode iterates over each tile. In that sense, this operation can be viewed as a kind of `gather` operation or as simply a permutation on the rows and cols.

Note that the first mode of each mode of the result is the sublayout `(3,(2,4)):(177,(13,2))` and is precisely the result we would have received if we had applied `composition` instead of `logical_divide`.
