---
title: "Partitioning a Tensor"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#partitioning-a-tensor"
---

## [Partitioning a Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#partitioning-a-tensor)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#partitioning-a-tensor "Permalink to this headline")

To implement generic partitioning of a `Tensor`, we apply composition or tiling followed by a slicing. This can be performed in many ways, but we have found three ways that are particularly useful: inner-partitioning, outer-partitioning, and TV-layout-partitioning.
