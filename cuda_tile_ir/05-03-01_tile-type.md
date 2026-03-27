---
title: "5.3.1. Tile Type"
section: "5.3.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#tile-type"
---

### [5.3.1. Tile Type](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-type)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-type "Permalink to this headline")

A tile is a tensor with static shape, i.e., the extent across each dimension
is known at compile time. Each extent must be a power of two.

> **Note**
>
> In **Tile IR**, all data values to be operated on are expressed as a tile.
> In particular, even scalar values are represented as a tile of rank zero.

We use `tile<MxNxKxE>` where `M`, `N`, `K` are integer numbers
and `E` is an element type as syntax for tile types, see
[Syntax](https://docs.nvidia.com/cuda/tile-ir/latest/sections/syntax.html#section-syntax) for more details.

| Example Tile Type | Description |
| --- | --- |
| `tile<2x8xf32>` | a 2-dimensional tensor with 2 rows and 8 columns of f32 values |
| `tile<f32>` | a single value of type f32 |
| `tile<ptr<f32>>` | a pointer to a value of type f32 |
| `tile<4x8xptr<f32>>` | a 2-dimensional tensor with 4 rows and 8 columns of pointers to values of type f32 |

> **Note**
>
> A tile/tensor of pointers is typically used to load from or store to a
> batch of locations. The tile of pointers defines the shape of the values
> that are loaded or stored, with one pointer element mapping to one
> scalar value loaded or stored. It does not imply any structure on the
> locations themselves. For example, two consecutive pointers in the
> tile may not point to consecutive locations in memory. Even more, the
> same location may be present multiple times within a single tile of
> pointers. See [Memory Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model) for a discussion of
> implications.
