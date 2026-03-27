---
title: "5.4.5.2.11. __nv_atomic_store(), __nv_atomic_store_n()"
section: "5.4.5.2.11"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-store-nv-atomic-store-n"
---

#### [5.4.5.2.11. __nv_atomic_store(), __nv_atomic_store_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-store-nv-atomic-store-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-store-nv-atomic-store-n "Permalink to this headline")

```cuda
__device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. | `__nv_atomic_store`  reads the value where `val` points to and stores to where `address` points to.
| `__nv_atomic_store_n` stores `val` to where `address` points to.

`order` cannot be `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.
