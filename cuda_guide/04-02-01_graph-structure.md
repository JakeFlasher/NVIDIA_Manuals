---
title: "4.2.1. Graph Structure"
section: "4.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-structure"
---

## [4.2.1. Graph Structure](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#graph-structure)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#graph-structure "Permalink to this headline")

An operation forms a node in a graph. The dependencies between the operations are the edges. These dependencies constrain the execution sequence of the operations.

An operation may be scheduled at any time once the nodes on which it depends are complete. Scheduling is left up to the CUDA system.
