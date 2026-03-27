---
title: "4.3.3.4.2. Set Access in the Importing Process"
section: "4.3.3.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#set-access-in-the-importing-process"
---

#### [4.3.3.4.2. Set Access in the Importing Process](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#set-access-in-the-importing-process)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#set-access-in-the-importing-process "Permalink to this headline")

Imported memory pools are initially only accessible from their resident
device. The imported memory pool does not inherit any accessibility set by the
exporting process. The importing process needs to enable access with
`cudaMemPoolSetAccess` from any GPU it plans to access the memory from.

If the imported memory pool belongs to a device that is not visible to importing
process, the user must use the `cudaMemPoolSetAccess` API to enable access
from the GPUs the allocations will be used on.
(See [Device Accessibility for Multi-GPU Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#stream-ordered-deviceaccessibility))
