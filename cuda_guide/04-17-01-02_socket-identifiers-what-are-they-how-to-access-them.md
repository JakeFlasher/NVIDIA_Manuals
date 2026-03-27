---
title: "4.17.1.2. Socket Identifiers: What are they? How to access them?"
section: "4.17.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#socket-identifiers-what-are-they-how-to-access-them"
---

### [4.17.1.2. Socket Identifiers: What are they? How to access them?](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#socket-identifiers-what-are-they-how-to-access-them)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#socket-identifiers-what-are-they-how-to-access-them "Permalink to this headline")

NUMA (Non-Uniform Memory Access) is a memory architecture used in
multi-processor computer systems such that the memory is divided into
multiple nodes. Each node has its own processors and memory. In such a
system, NUMA divides the system into nodes and assigns a unique
identifier (*numaID*) to every node.

EGM uses the NUMA node identifier which is assigned by the operating
system. Note that, this identifier is different from the ordinal of a
device and it is associated with the closest host node. In addition to
the existing methods, the user can obtain the identifier of the host
node (*numaID*) by calling [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)
with `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` attribute type as follows:

```c++
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);
```
