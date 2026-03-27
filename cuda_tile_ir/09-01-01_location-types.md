---
title: "9.1.1. Location Types"
section: "9.1.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#location-types"
---

### [9.1.1. Location Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#location-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#location-types "Permalink to this headline")

Two location types are supported by **Tile IR**:

1. A **Tile IR** location represented by a `#cuda_tile.di_loc` attribute which combines file, line, column information with scope metadata as shown in the example above. This is the preferred method for generating location information fused with scope metadata.
2. A `CallSiteLoc` where both the callee and caller are supported location types. This respresents a function call e.g. in the case where the **Tile IR** producer supports inlining.

All **Tile IR** global variables will have `UnknownLoc` given that `#cuda_tile.di_loc` supports local scope only and there is no support for `FileLineColLoc`.

| Location Type | Support |
| --- | --- |
| CudaTile DILocAttr | Supported |
| CallSiteLoc | Supported if BOTH callee and caller are supported. |
| FusedLoc | Unsupported |
| NameLoc | Unsupported |
| OpaqueLoc | Unsupported |
| FileLineColLoc | Supported only as part of DILocAttr, else converted to UnknownLoc. |
| UnknownLoc | Supported |
