---
title: "5.7.1.1. Scope Relationships"
section: "5.7.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#scope-relationships"
---

### [5.7.1.1. Scope Relationships](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#scope-relationships)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#scope-relationships "Permalink to this headline")

**Each program thread is related to each other program thread by one or more thread scope relations:**

  - Each thread in the system is related to each other thread in the system by the _system_ thread scope:
  `cuda::thread_scope_system`.
  - Each GPU thread is related to each other GPU thread in the same CUDA device and within the same
  [memory synchronization domain](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html#memory-synchronization-domains)
  by the _device_ thread scope: `cuda::thread_scope_device`.
  - Each GPU thread is related to each other GPU thread in the same CUDA thread block by the _block_ thread scope:
  `cuda::thread_scope_block`.
  - Each thread is related to itself by the _thread_ thread scope: `cuda::thread_scope_thread`.
