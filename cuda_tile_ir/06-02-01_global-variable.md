---
title: "6.2.1. Global Variable"
section: "6.2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#global-variable"
---

### [6.2.1. Global Variable](https://docs.nvidia.com/cuda/tile-ir/latest/sections#global-variable)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#global-variable "Permalink to this headline")

A global is a named global variable that is stored in global device memory and accessible to all tile blocks.
Global variables are declared using the [cuda_tile.global](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-global) operation.
A global variable must be initialized upon declaration and will be initialized exactly once.

A global variable must contain a value of [Tile Type](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile).

A global variable can be modified by using the [cuda_tile.get_global](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-global) operation to obtain a pointer which
can be used to read and write to the global variable.
