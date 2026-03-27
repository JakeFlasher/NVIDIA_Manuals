---
title: "5.4.5.2.5. __nv_atomic_fetch_xor(), __nv_atomic_xor()"
section: "5.4.5.2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-fetch-xor-nv-atomic-xor"
---

#### [5.4.5.2.5. __nv_atomic_fetch_xor(), __nv_atomic_xor()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-xor-nv-atomic-xor)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-xor-nv-atomic-xor "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old ^ val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_xor` returns the `old` value.
- `__nv_atomic_xor` has no return value.

The functions support the following data types:

- Any integral type of size 4 or 8 bytes.
