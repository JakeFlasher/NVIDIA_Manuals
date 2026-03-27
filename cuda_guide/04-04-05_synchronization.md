---
title: "4.4.5. Synchronization"
section: "4.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#synchronization"
---

## [4.4.5. Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#synchronization)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#synchronization "Permalink to this headline")

Prior to the introduction of Cooperative Groups, the CUDA programming model only allowed synchronization between thread blocks at a kernel completion boundary.
Cooperative groups allows developers to synchronize groups of cooperating threads at different granularities.
