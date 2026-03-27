---
title: "6.8.1. Initialization"
section: "6.8.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#initialization"
---

### [6.8.1. Initialization](https://docs.nvidia.com/cuda/tile-ir/latest/sections#initialization)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#initialization "Permalink to this headline")

A launch of tile kernel initializes the abstract machine with:

- The module representing the complete program.
- A grid of tile blocks where each tile block is instantiated using the same tile kernel, begins at statement 0,
assigned a unique grid coordinate, and assigned a unique empty register file.
- A reference to global memory, where its state is the state of global memory prior to the kernel launch.
- The set of pending memory operations is initialize to be empty.
