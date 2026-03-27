---
title: "6.8.2. Forward Progress"
section: "6.8.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#forward-progress"
---

### [6.8.2. Forward Progress](https://docs.nvidia.com/cuda/tile-ir/latest/sections#forward-progress)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#forward-progress "Permalink to this headline")

Execution proceeds with unspecified scheduling of tile blocks. Each tile block will be executed in some order
which is non-deterministic and not specified by the language semantics. We guarantee forward progress of the execution
that is all tile blocks will be guaranteed to eventually be scheduled for execution. It is possible that all tile
blocks run completely in parallel, completely serially, or anything in between.
