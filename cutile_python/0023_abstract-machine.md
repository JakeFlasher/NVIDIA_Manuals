---
title: "Abstract Machine"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#abstract-machine"
---

## [Abstract Machine](https://docs.nvidia.com/cuda/cutile-python#abstract-machine)[](https://docs.nvidia.com/cuda/cutile-python/#abstract-machine "Permalink to this headline")

A _tile kernel_ is executed by logical thread [blocks](https://docs.nvidia.com/cuda/cutile-python/#block) that are organized in
a 1D, 2D, or 3D _grid_.

Each _block_ is executed by a subset of a GPU, which is decided by the
implementation, not the programmer.
Each [block](https://docs.nvidia.com/cuda/cutile-python/#block) executes the body of the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels).
Scalar operations are executed serially by a single thread of the [block](https://docs.nvidia.com/cuda/cutile-python/#block),
and array operations are collectively executed in parallel by all threads of
the [block](https://docs.nvidia.com/cuda/cutile-python/#block).

Tile programs explicitly describe [block](https://docs.nvidia.com/cuda/cutile-python/#block)-level parallelism, but not
thread-level parallelism.
Threads cannot be explicitly identified or manipulated in tile programs.

Explicit synchronization or communication within a [block](https://docs.nvidia.com/cuda/cutile-python/#block) is not
permitted, but it is allowed between different [blocks](https://docs.nvidia.com/cuda/cutile-python/#block).

It is important to not confuse [blocks](https://docs.nvidia.com/cuda/cutile-python/#block) (units of execution) with
[tiles](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars) (units of data).
A block may work with multiple different [tiles](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars) with
differing shapes originating from differing [global arrays](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array).
