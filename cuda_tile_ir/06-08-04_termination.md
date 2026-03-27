---
title: "6.8.4. Termination"
section: "6.8.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#termination"
---

### [6.8.4. Termination](https://docs.nvidia.com/cuda/tile-ir/latest/sections#termination)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#termination "Permalink to this headline")

A tile block will terminate when the tile block’s function body reaches the final statement. Tile kernels must
terminate with a return operation [cuda_tile.return](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-return) which signals the end of the execution.
