---
title: "4.4.8. Large Scale Groups"
section: "4.4.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#large-scale-groups"
---

## [4.4.8. Large Scale Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#large-scale-groups)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#large-scale-groups "Permalink to this headline")

Cooperative Groups allows for large groups that span the entire grid.
All Cooperative Group functionality described previously is available to these large groups, with one notable exception:
synchronizing the entire grid requires using the `cudaLaunchCooperativeKernel` runtime launch API.

Multi-device launch APIs and related references for Cooperative Groups have been removed as of CUDA 13.
