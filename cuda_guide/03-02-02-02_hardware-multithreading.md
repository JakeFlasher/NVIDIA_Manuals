---
title: "3.2.2.2. Hardware Multithreading"
section: "3.2.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading"
---

### [3.2.2.2. Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#hardware-multithreading)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#hardware-multithreading "Permalink to this headline")

When an SM is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled for execution by a _warp scheduler_. The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0. [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-thread-hierarchy-review) describes how thread IDs relate to thread indices in the block.

The total number of warps in a block is defined as follows:

\(\(\text{ceil}\left( \frac{T}{W_{size}}, 1 \right)\)\)

- _T_ is the number of threads per block,
- _Wsize_ is the warp size, which is equal to 32,
- ceil(x, y) is equal to x rounded up to the nearest multiple of y.

![A thread block is partitioned into warps of 32 threads.](images/____w___-_______________1.png)

Figure 19 A thread block is partitioned into warps of 32 threads.[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#hardware-multithreading-warps-in-a-block "Link to this image")

The execution context (program counters, registers, etc.) for each warp processed by an SM is maintained on-chip throughout the warp’s lifetime. Therefore, switching between warps incurs no cost. At each instruction issue cycle, a warp scheduler selects a warp with threads ready to execute its next instruction (the [active threads](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#simt-architecture-notes) of the warp) and issues the instruction to those threads.

Each SM has a set of 32-bit registers that are partitioned among the warps, and a [shared memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory) that is partitioned among the thread blocks. The number of blocks and warps that can reside and be processed concurrently on the SM for a given kernel depends on the amount of registers and shared memory used by the kernel, as well as the amount of registers and shared memory available on the SM. There are also a maximum number of resident blocks and warps per SM. These limits, as well the amount of registers and shared memory available on the SM, depend on the compute capability of the device and are specified in [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities). If there are not enough resources available per SM to process at least one block, the kernel will fail to launch. The total number of registers and shared memory allocated for a block can be determined in several ways documented in the [Occupancy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy) section.
