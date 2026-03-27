---
title: "Description"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#description"
---

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#description)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#description "Permalink to this headline")

The `broadcast` operation expands each unary (`1`) dimension in the input tile
by duplicating the data along that dimension.

Expansion happens only for dimensions of size one that are stretched or “copied” to match
the size of the dimension implied by the result type of the operation. The operation
does not change the rank of the source tile.  Any change to the rank of the source tile
must be made using reshape-like operations before broadcasting.
