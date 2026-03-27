---
title: "Summary"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#03_tensor--summary"
---

## [Summary](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#summary)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#summary "Permalink to this headline")

- `Tensor` is defined as an `Engine` and a `Layout`.
  - `Engine` is an iterator that can be offset and dereferenced.
  - `Layout` defines the logical domain of the tensor and maps coordinates to offsets.
- Tile a `Tensor` using the same methods for tiling `Layout`s.
- Slice a `Tensor` to retrieve subtensors.
- Partitioning is tiling and/or composition followed by slicing.
