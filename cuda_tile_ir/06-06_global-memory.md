---
title: "6.6. Global Memory"
section: "6.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#global-memory"
---

## [6.6. Global Memory](https://docs.nvidia.com/cuda/tile-ir/latest/sections#global-memory)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#global-memory "Permalink to this headline")

The global memory, \(\(M\)\), is a mapping from addresses to scalar values.

The global memory is used to store the values of the tile block’s global variables.

The heap is abstractly modeled as map from addresses to scalar values, not tile values.
This distinction is essential to describe the memory effect of tile operations as a sequence of individual
scalar memory operations. A fine-grained model enables both reasoning about aggregate operations granularly
as well as a straight forward denotation into the existing PTX memory model.

> **Note**
>
> Global memory **is** the same global device memory that is used by CUDA programs
> and described in the [PTX Specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces).

The specification elaborates the intricacies of the **Tile IR** memory model in [Memory Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model).
