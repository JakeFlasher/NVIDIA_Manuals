---
title: "8.4. Conversions"
section: "8.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#conversions"
---

## [8.4. Conversions](https://docs.nvidia.com/cuda/tile-ir/latest/sections#conversions)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#conversions "Permalink to this headline")

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations
or rules for conversion.

[cuda_tile.bitcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-bitcast) preserves the contents of the input but allows for changing of element types, [cuda_tile.exti](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-exti) and [cuda_tile.trunci](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-trunci) change the width of integer tiles,
[cuda_tile.ftoi](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-ftoi) and [cuda_tile.itof](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-itof) convert floating-point tiles to integer tiles and vice versa, and [cuda_tile.ftof](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-ftof) converts between different floating-point types.

For more details on conversions and their rules see the individual operation’s documentation.
