---
title: "1.1. Scalable Data Parallel Computing with Tiles on GPU"
section: "1.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html#scalable-data-parallel-computing-with-tiles-on-gpu"
---

## [1.1. Scalable Data Parallel Computing with Tiles on GPU](https://docs.nvidia.com/cuda/tile-ir/latest/sections#scalable-data-parallel-computing-with-tiles-on-gpu)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#scalable-data-parallel-computing-with-tiles-on-gpu "Permalink to this headline")

The CUDA platform provides portability through CUDA C++ and PTX, allowing code written for older GPU generations to run
efficiently on newer ones. These mechanisms enable developers to write software for one family of GPUs and
continue to deploy it on future generations.

The rapid evolution of hardware features, such as tensor cores, has increased programming-model complexity,
requiring greater expertise and hardware understanding to write portable and performant code.
Since Volta, each new GPU generation introduces powerful new hardware features, giving rise to new programming paradigms.

To address these challenges, **Tile IR** introduces a virtual instruction set that enables native programming of the hardware in terms of tiles,
allowing developers to write higher-level code that can be efficiently executed across multiple
GPUs with minimal changes.

While PTX ensures portability for SIMT programs, **Tile IR** extends the CUDA platform with native support for tile-based programs,
allowing developers to focus on partitioning their data-parallel programs into tiles and tile blocks, letting **Tile IR** handle
the mapping onto hardware resources such as threads, the memory hierarchy, and tensor cores.

By raising the level of abstraction, **Tile IR** enables users to build new higher-level hardware-specific DSLs, compilers,
and frameworks for NVIDIA hardware.
