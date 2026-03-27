---
title: "2.1. Tile Kernels"
section: "2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#tile-kernels"
---

## [2.1. Tile Kernels](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-kernels)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-kernels "Permalink to this headline")

**Tile IR** programs are referred to as _tile kernels_, which like CUDA C++ or PTX, are functions which run as *N*
copies in parallel when invoked. The primary difference is the basic unit of execution: a _tile-block_, which
expresses the computation performed by a single logical tile thread operating over a multi-dimensional
tile of data.

During execution, each tile kernel is referred to as a tile kernel instance.

Below is a simple **Tile IR** kernel which prints “Hello World!”.

```mlir
cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}
```

For those familiar with CUDA threads, it is important to note that **Tile IR**’s threads are different.
Before we go any further into formalisms, it is essential we highlight those differences.

Tile kernels are the entry point of the program, executing as parallel instances of tile blocks.
