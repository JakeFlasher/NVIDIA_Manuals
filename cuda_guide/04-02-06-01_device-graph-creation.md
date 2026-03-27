---
title: "4.2.6.1. Device Graph Creation"
section: "4.2.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-creation"
---

### [4.2.6.1. Device Graph Creation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-creation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-creation "Permalink to this headline")

In order for a graph to be launched from the device, it must be instantiated explicitly for device launch. This is achieved by passing the `cudaGraphInstantiateFlagDeviceLaunch` flag to the `cudaGraphInstantiate()` call. As is the case for host graphs, device graph structure is fixed at time of instantiation and cannot be updated without re-instantiation, and instantiation can only be performed on the host. In order for a graph to be able to be instantiated for device launch, it must adhere to various requirements.
