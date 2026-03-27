---
title: "4.13.5. Reset L2 Access to Normal"
section: "4.13.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#reset-l2-access-to-normal"
---

## [4.13.5. Reset L2 Access to Normal](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#reset-l2-access-to-normal)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#reset-l2-access-to-normal "Permalink to this headline")

A persisting L2 cache line from a previous CUDA kernel may persist in L2 long after it has been used. Hence, a reset to normal for L2 cache is important for streaming or normal memory accesses to utilize the L2 cache with normal priority. There are three ways a persisting access can be reset to normal status.

1. Reset a previous persisting memory region with the access property, `cudaAccessPropertyNormal`.
2. Reset all persisting L2 cache lines to normal by calling `cudaCtxResetPersistingL2Cache()`.
3. **Eventually** untouched lines are automatically reset to normal. Reliance on automatic reset is strongly discouraged because of the undetermined length of time required for automatic reset to occur.
