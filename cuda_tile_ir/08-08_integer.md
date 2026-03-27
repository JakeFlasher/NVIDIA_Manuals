---
title: "8.8. Integer"
section: "8.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#integer"
---

## [8.8. Integer](https://docs.nvidia.com/cuda/tile-ir/latest/sections#integer)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#integer "Permalink to this headline")

**Tile IR** contains a set of typed arithmetic operations which implement familiar arithmetic operations on tiles of integers, for floating-point
operations see [Floating Point](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-group-floating-point).

All operations are implemented in a manner that is efficient for the target architecture and device family. In most common cases this means
utilizing the underlying hardware’s native floating-point operations. Due to **Tile IR**’s stability guarantees and higher-level programming
model some types on some hardware may be emulated, see [Stability](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#section-stability) for more information about the stability guarantees and information
about per device behavior.
