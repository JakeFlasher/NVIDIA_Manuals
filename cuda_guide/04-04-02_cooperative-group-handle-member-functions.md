---
title: "4.4.2. Cooperative Group Handle & Member Functions"
section: "4.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-group-handle-member-functions"
---

## [4.4.2. Cooperative Group Handle & Member Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cooperative-group-handle-member-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cooperative-group-handle-member-functions "Permalink to this headline")

Cooperative Groups are managed via a Cooperative Group Handle.
The Cooperative Group handle allows participating threads to learn their position in the group, the group size, and other group information.
Select group member functions are shown in the following table.

| Accessor | Returns |
| --- | --- |
| `thread_rank()` | The rank of the calling thread. |
| `num_threads()` | The total number of threads in the group  . |
| `thread_index()` | A 3-Dimensional index of the thread within the launched block. |
| `dim_threads()` | The 3D dimensions of the launched block in units of threads. |

A complete list pf member functions is available in the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cg-api-common-header).
