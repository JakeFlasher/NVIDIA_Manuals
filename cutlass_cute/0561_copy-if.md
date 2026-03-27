---
title: "copy_if"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#copy-if"
---

## [copy_if](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#copy-if)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#copy-if "Permalink to this headline")

CuTe’s `copy_if` algorithm lives in the same header as `copy`,
[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp).
The algorithm takes source and destination `Tensor` parameters like `copy`,
but it also takes a “predication `Tensor`”
with the same shape as the input and output.
Elements of the source `Tensor` are only copied
if the corresponding predication `Tensor` element is nonzero.

For details on why and how to use `copy_if`,
please refer to the
[“predication” section of the tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html).
