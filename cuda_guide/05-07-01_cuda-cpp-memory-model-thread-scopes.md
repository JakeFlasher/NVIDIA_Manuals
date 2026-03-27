---
title: "5.7.1. Thread Scopes"
section: "5.7.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#cuda-cpp-memory-model--thread-scopes"
---

## [5.7.1. Thread Scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#thread-scopes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#thread-scopes "Permalink to this headline")

A **thread scope** specifies the kind of threads that can synchronize with each other using a synchronization primitive such
as [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) or [cuda::barrier](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html).

```cuda
namespace cuda {

enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};

}  // namespace cuda
```
