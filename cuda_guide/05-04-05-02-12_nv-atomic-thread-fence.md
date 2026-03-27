---
title: "5.4.5.2.12. __nv_atomic_thread_fence()"
section: "5.4.5.2.12"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-thread-fence"
---

#### [5.4.5.2.12. __nv_atomic_thread_fence()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-thread-fence)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-thread-fence "Permalink to this headline")

```cuda
__device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

This atomic function establishes an ordering between memory accesses requested by this thread based on the specified memory order. The thread scope parameter specifies the set of threads that may observe the ordering effect of this operation.
