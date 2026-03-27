---
title: "Logical Divide 1-D Example"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#logical-divide-1-d-example"
---

### [Logical Divide 1-D Example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#logical-divide-1-d-example)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#logical-divide-1-d-example "Permalink to this headline")

Consider tiling the 1-D layout `A = (4,2,3):(2,1,8)` with the tiler `B = 4:2`. Informally, this means that we have a 1-D vector of 24 elements in some storage order defined by `A` and we want to extract tiles of 4 elements strided by 2.

This is computed in the three steps described in the implementation above.

- Complement of `B = 4:2` under `size(A) = 24` is `B* = (2,3):(1,8)`.
- Concantenation of `(B,B*) = (4,(2,3)):(2,(1,8))`.
- Composition of `A = (4,2,3):(2,1,8)` with `(B,B*)` is then `((2,2),(2,3)):((4,1),(2,8))`.

![divide1.png](images/_______-______-_-_-________1.png)

The above figure depicts `A` as a 1-D layout with the elements pointed to by `B` highlighted in gray. The layout `B` describes our “tile” of data, and there are six of those tiles in `A` shown by each of the colors. After the divide, the first mode of the result is the tile of data and the second mode of the result iterates over each tile.
