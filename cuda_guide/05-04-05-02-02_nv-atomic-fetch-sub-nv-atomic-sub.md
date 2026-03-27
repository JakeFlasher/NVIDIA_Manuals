---
title: "5.4.5.2.2. __nv_atomic_fetch_sub(), __nv_atomic_sub()"
section: "5.4.5.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-fetch-sub-nv-atomic-sub"
---

#### [5.4.5.2.2. __nv_atomic_fetch_sub(), __nv_atomic_sub()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-sub-nv-atomic-sub)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-sub-nv-atomic-sub "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old - val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_sub` returns the `old` value.
- `__nv_atomic_sub` has no return value.

The functions support the following data types:

- `int`, `unsigned`, `unsigned long long`, `float`, `double`.
