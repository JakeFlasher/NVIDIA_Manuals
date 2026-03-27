---
title: "5.1. Element Types"
section: "5.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#element-types"
---

## [5.1. Element Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#element-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#element-types "Permalink to this headline")

Element types are the native data types supported by **Tile IR**. By
themselves they do not describe a value. As **Tile IR** operates over
tensors, these types describe the hardware accelerated, primitive values
that can be contained by a tensor. They specify how a sequence of bits
are to be interpreted. Each element type has a size associated with it
that represents the number of bits required to represent it.

> **Note**
>
> Note that this is different from a potential storage size, which is
> specified by the data layout of the tensor which contains these
> values.

Element types come in two flavors, general purpose fundamental types
that come without restriction, and specialized alternative types which
each come with a set of restrictions.
