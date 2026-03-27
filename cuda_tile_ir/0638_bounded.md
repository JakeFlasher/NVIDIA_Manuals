---
title: "Bounded"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#bounded"
---

#### [Bounded](https://docs.nvidia.com/cuda/tile-ir/latest/sections#bounded)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#bounded "Permalink to this headline")

```mlir
#bounded<(lb|?), (ub|?)>
```

The `bounded` attribute must be used as a predicate for
`cuda_tile.assume`. The predicated value must be a tile of integers.

`bounded` specifies a lower and upper bound for all elements of the
predicated tile when interpreted as signed integers. Bounds are optional:
it is possible to leave a bound unspecified, as indicated by “?” in the
assembly format. E.g., `#bounded<0, ?>`. Both lower bound and upper
bound are inclusive.

The lower bounds must be less than or equal to the upper bound. A lower/
upper bound that exceeds the range of valid values of the predicated value
is invalid.

```mlir
%1 = cuda_tile.assume #cuda_tile.bounded<0, ?>, %0
    : !cuda_tile.tile<4x8xi16>
```
