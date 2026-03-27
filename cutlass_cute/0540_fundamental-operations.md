---
title: "Fundamental operations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#fundamental-operations"
---

## [Fundamental operations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#fundamental-operations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#fundamental-operations "Permalink to this headline")

CuTe `Tensor` provides container-like operations for accessing elements.

- `.data()`. The iterator this `Tensor` holds.
- `.size()`. The total logical size of this `Tensor`.
- `.operator[](Coord)`. Access the element corresponding to the logical coordinate `Coord`.
- `.operator()(Coord)`. Access the element corresponding to the logical coordinate `Coord`.
- `.operator()(Coords...)`. Access the element corresponding to the logical coordinate `make_coord(Coords...)`.

CuTe `Tensor` provides a similar core of hierarchical operations as `Layout`.

- `rank<I...>(Tensor)`. The rank of the `I...`th mode of the `Tensor`.
- `depth<I...>(Tensor)`. The depth of the `I...`th mode of the `Tensor`.
- `shape<I...>(Tensor)`. The shape of the `I...`th mode of the `Tensor`.
- `size<I...>(Tensor)`. The size of the `I...`th mode of the `Tensor`.
- `layout<I...>(Tensor)`. The layout of the `I...`th mode of the `Tensor`.
- `tensor<I...>(Tensor)`. The subtensor corresponding to the the `I...`th mode of the `Tensor`.
