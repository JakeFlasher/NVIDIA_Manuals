---
title: "7. Memory Model"
section: "7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#memory-model"
---

# [7. Memory Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections#memory-model)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#memory-model "Permalink to this headline")

The memory model defines the legal values that loads can return from memory.
This is not as straightforward as one might expect at first glance, to enable compiler and hardware optimizations, we allow the apparent re-ordering of instructions.

This memory model is derived from the PTX memory model, and synchronization primitives are deliberately similar.
