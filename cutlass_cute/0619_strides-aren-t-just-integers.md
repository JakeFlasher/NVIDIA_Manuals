---
title: "Strides aren’t just integers"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#strides-aren-t-just-integers"
---

### [Strides aren’t just integers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#strides-aren-t-just-integers)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#strides-aren-t-just-integers "Permalink to this headline")

Ordinary tensors have a layout that maps
a logical coordinate `(i,j)` into a 1-D linear index `k`.
This mapping is the inner-product of the coordinate with the strides.

TMA Tensors hold iterators of TMA coordinates.
Thus, a TMA Tensor’s Layout must map a logical coordinate
to a TMA coordinate, rather than to a 1-D linear index.

To do this, we can abstract what a stride is. Strides need not be integers, but rather any algebraic object that supports inner-product with the integers (the logical coordinate). The obvious choice is the `ArithmeticTuple` we used earlier since they can be added to each other, but this time additionally equipped with an `operator*` so it can also be scaled by an integer.
