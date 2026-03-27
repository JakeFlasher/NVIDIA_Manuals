---
title: "7.9. No-thin-air"
section: "7.9"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#no-thin-air"
---

## [7.9. No-thin-air](https://docs.nvidia.com/cuda/tile-ir/latest/sections#no-thin-air)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#no-thin-air "Permalink to this headline")

It is not practical to specify a “no thin air” axiom without preventing useful compiler optimizations.
We therefore say informally that the implementation will not provide values out of thin air to satisfy program executions.
