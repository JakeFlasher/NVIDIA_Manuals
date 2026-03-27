---
title: "Overview of CUDA’s synchronization methods"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#overview-of-cuda-s-synchronization-methods"
---

## [Overview of CUDA’s synchronization methods](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#overview-of-cuda-s-synchronization-methods)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#overview-of-cuda-s-synchronization-methods "Permalink to this headline")

The CUDA programming model provides 3 abstractions:

- hierarchical parallelism – that is, parallel threads
grouped into hierarchical units such as blocks and clusters;
- shared memory, through which parallel threads that are
in the same hierarchical unit can communicate; and
- synchronization methods for threads.

These abstractions help developers extract
both fine-grained and coarse-grained parallelism,
by making it possible for them to subdivide problems
into independent components,
and to insert synchronization at appropriate points.

Over the years CUDA has introduced several synchronization primitives
that operate at different levels of the hierarchy.
These include

- [thread block - level](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) synchronization (e.g., `__syncthreads()`);
- [warp-level](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) synchronization (e.g., `__syncwarp()`); and
- [thread-level](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions) fence operations.

As an extension to this, starting with the Hopper architecture, CUDA added the following improvements:

- [thread block clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters) –
a new level in the thread hierarchy representing
a group of thread blocks that can coordinate and share data;
- synchronization instructions for a thread block cluster and threads within a cluster scope.
