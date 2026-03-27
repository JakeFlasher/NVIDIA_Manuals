---
title: "9.1.3. Options"
section: "9.1.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#options"
---

### [9.1.3. Options](https://docs.nvidia.com/cuda/tile-ir/latest/sections#options)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#options "Permalink to this headline")

**Tile IR** producers have two options for generating location information fused with scope metadata:

1. The **Tile IR** producer can generate location information fused with scope metadata directly using `#cuda_tile.di_loc` as shown in the example above.
2. The **Tile IR** producer can generate a `loc` with file, line, column location information and then use a **Tile IR** pass (via `--synthesize-debug-info-scopes`) to synthesize scope metadata from that file, line, column location information.

> **Note**
>
> File, line, column location information must be converted to a **Tile IR** location prior to bytecode generation as all `FileLineColLoc` attributes are encoded as `UnknownLoc` attributes in the bytecode.
