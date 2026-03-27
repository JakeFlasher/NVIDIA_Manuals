---
title: "4.17.2.2. Single-Node, Multi-GPU"
section: "4.17.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#single-node-multi-gpu"
---

### [4.17.2.2. Single-Node, Multi-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#single-node-multi-gpu)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#single-node-multi-gpu "Permalink to this headline")

In a multi-GPU system, the user has to provide host information for
the placement. As we mentioned, a natural way to express that
information would be by using NUMA node IDs and EGM follows this
approach. Therefore, using the `cuDeviceGetAttribute` function the
user should be able to learn the closest NUMA node id. (See [Socket Identifiers: What are they? How to access them?](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#socket-identifiers-what-are-they-how-to-access-them)).
Then the user can allocate and manage EGM memory using VMM (Virtual
Memory Management) API or CUDA Memory Pool.
