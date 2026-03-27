---
title: "5.4.5.2.1. __nv_atomic_fetch_add(), __nv_atomic_add()"
section: "5.4.5.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-fetch-add-nv-atomic-add"
---

#### [5.4.5.2.1. __nv_atomic_fetch_add(), __nv_atomic_add()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-add-nv-atomic-add)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-add-nv-atomic-add "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old + val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_add` returns the `old` value.
- `__nv_atomic_add` has no return value.

The functions support the following data types:

- `int`, `unsigned`, `unsigned long long`, `float`, `double`.
