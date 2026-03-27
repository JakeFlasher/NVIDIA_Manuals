---
title: "7.10. Data Races"
section: "7.10"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#data-races"
---

## [7.10. Data Races](https://docs.nvidia.com/cuda/tile-ir/latest/sections#data-races)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#data-races "Permalink to this headline")

Two accesses are said to _conflict_ when they access the same location and at least one of them is a write.

Two conflicting memory accesses are said to be in a _data race_ if they are not related in happens before and are they are not morally strong.

Programs with data races have **undefined behaviour**.
