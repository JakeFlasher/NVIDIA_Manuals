---
title: "5.7.2. Synchronization primitives"
section: "5.7.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#cuda-cpp-memory-model--synchronization-primitives"
---

## [5.7.2. Synchronization primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#synchronization-primitives)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#synchronization-primitives "Permalink to this headline")

Types in namespaces `std::` and `cuda::std::` have the same behavior as corresponding types in namespace `cuda::`
when instantiated with a scope of `cuda::thread_scope_system`.
