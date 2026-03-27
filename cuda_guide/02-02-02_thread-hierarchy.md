---
title: "2.2.2. Thread Hierarchy"
section: "2.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#thread-hierarchy"
---

## [2.2.2. Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#thread-hierarchy)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#thread-hierarchy "Permalink to this headline")

Threads are organized into thread blocks, which are then organized into a grid.  Grids may be 1, 2, or 3 dimensional and the size of the grid can be queried inside a kernel with the `gridDim` built-in variable.  Thread blocks may also be 1, 2, or 3 dimensional. The size of the thread block can be queried inside a kernel with the `blockDim` built-in variable. The index of the thread block can be queried with the `blockIdx` built-in variable.  Within a thread block, the index of the thread is obtained using the `threadIdx` built-in variable.  These built-in variables are used to compute a unique global thread index for each thread, thereby enabling each thread to load/store specific data from global memory and execute a unique code path as needed.

- `gridDim.[x|y|z]`: Size of the grid in the `x`, `y` and `z` dimension respectively.  These values are set at kernel launch.
- `blockDim.[x|y|z]`: Size of the block in the `x`, `y` and `z` dimension respectively.  These values are set at kernel launch.
- `blockIdx.[x|y|z]`: Index of the block in the `x`, `y` and `z` dimension respectively.  These values change depending on which block is executing.
- `threadIdx.[x|y|z]`: Index of the thread in the `x`, `y` and `z` dimension respectively.  These values change depending on which thread is executing.

The use of multi-dimensional thread blocks and grids is for convenience only and does not affect performance. The threads of a block are linearized predictably: the first index `x` moves the fastest, followed by `y` and then `z`.  This means that in the linearization of a thread indices, consecutive values of `threadIdx.x` indicate consecutive threads, `threadIdx.y` has a stride of `blockDim.x`, and `threadIdx.z` has a stride of  `blockDim.x * blockDim.y`. This affects how threads are assigned to warps, as detailed in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-hardware-multithreading).

[Figure 9](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-thread-hierarchy-review-grid-of-thread-blocks) shows a simple example of a 2D grid, with 1D thread blocks.

![Grid of Thread Blocks](images/______-__________1.png)

Figure 9 Grid of Thread Blocks[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#writing-cuda-kernels-thread-hierarchy-review-grid-of-thread-blocks "Link to this image")
