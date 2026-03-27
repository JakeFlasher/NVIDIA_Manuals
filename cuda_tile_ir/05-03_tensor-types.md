---
title: "5.3. Tensor Types"
section: "5.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#tensor-types"
---

## [5.3. Tensor Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tensor-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tensor-types "Permalink to this headline")

A tensor is a multi-dimensional, rectangular array described by a shape
and element type. The shape is a vector that describes the number of
elements across each axis of the tensor. The length of said vector
describes the rank of the tensor, i.e., the number of its dimensions.
All tensors in **Tile IR** have a statically known rank. **Tile IR** has
two kinds of tensor types tiles and views.
