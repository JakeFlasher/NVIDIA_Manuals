---
title: "5.4.5.2.6. __nv_atomic_fetch_min(), __nv_atomic_min()"
section: "5.4.5.2.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-fetch-min-nv-atomic-min"
---

#### [5.4.5.2.6. __nv_atomic_fetch_min(), __nv_atomic_min()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-min-nv-atomic-min)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-min-nv-atomic-min "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the minimum of `old` and `val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_min` returns the `old` value.
- `__nv_atomic_min` has no return value.

The functions support the following data types:

- `unsigned`, `int`, `unsigned long long`, `long long`.
