---
title: "7.4. Moral Strength"
section: "7.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#moral-strength"
---

## [7.4. Moral Strength](https://docs.nvidia.com/cuda/tile-ir/latest/sections#moral-strength)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#moral-strength "Permalink to this headline")

Two accesses to the same location are morally strong if the operations are related in _restricted program order_, or each operation specifies a scope which includes the tile block executing the other operation.
