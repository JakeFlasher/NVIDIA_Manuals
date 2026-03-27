---
title: "1.2.2.1.1. Thread Block Clusters"
section: "1.2.2.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters"
---

#### [1.2.2.1.1. Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction#thread-block-clusters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#thread-block-clusters "Permalink to this headline")

In addition to thread blocks, GPUs with compute capability 9.0 and higher have an optional level of grouping called _clusters_. Clusters are a group of thread blocks which, like thread blocks and grids, can be laid out in 1, 2, or 3 dimensions. [Figure 5](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#figure-thread-block-clusters) illustrates a grid of thread blocks that is also organized into clusters. Specifying clusters does not change the grid dimensions or the indices of a thread block within a grid.

![Thread blocks scheduled on SMs](images/______-_____-_________1.png)

Figure 5 When clusters are specified, thread blocks are in the same location in the grid but also have a position within the containing cluster.[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#figure-thread-block-clusters "Link to this image")

Specifying clusters groups adjacent thread blocks into clusters and provides some additional opportunities for synchronization and communication at the cluster level. Specifically, all thread blocks in a cluster are executed in a single GPC. [Figure 6](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#thread-block-scheduling-with-clusters) shows how thread blocks are scheduled to SMs in a GPC when clusters are specified. Because the thread blocks are scheduled simultaneously and within a single GPC, threads in different blocks but within the same cluster can communicate and synchronize with each other using software interfaces provided by [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-cooperative-groups). Threads in clusters can access the shared memory of all blocks in the cluster, which is referred to as [distributed shared memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory).The maximum size of a cluster is hardware dependent and varies between devices.

[Figure 6](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#thread-block-scheduling-with-clusters) illustrates the how thread blocks within a cluster are scheduled simultaneously on SMs within a GPC. Thread blocks within a cluster are always adjacent to each other within the grid.

![Thread blocks scheduled in clusters on GPCs](images/______-_____-_________2.png)

Figure 6 When clusters are specified, the thread blocks in a cluster are arranged in their cluster shape within the grid. The thread blocks of a cluster are scheduled simultaneously on the SMs of a single GPC.[](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/#thread-block-scheduling-with-clusters "Link to this image")
