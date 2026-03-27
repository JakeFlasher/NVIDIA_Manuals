---
title: "4.18.2.1. Parent and Child Grids"
section: "4.18.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#parent-and-child-grids"
---

### [4.18.2.1. Parent and Child Grids](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#parent-and-child-grids)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#parent-and-child-grids "Permalink to this headline")

A device thread that configures and launches a new grid belongs to the parent grid. The new grid that is created by the invocation is called a child grid.

The invocation and completion of child grids is properly nested, meaning that the parent grid is not considered complete until all child grids created by its threads have completed, and the runtime guarantees an implicit synchronization between the parent and child.

![Parent-Child Launch Nesting](images/______-___-_____-______1.png)

Figure 54 Parent-Child Launch Nesting[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#parent-child-launch-nesting-figure "Link to this image")
