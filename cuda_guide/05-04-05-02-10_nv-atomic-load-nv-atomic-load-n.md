---
title: "5.4.5.2.10. __nv_atomic_load(), __nv_atomic_load_n()"
section: "5.4.5.2.10"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-load-nv-atomic-load-n"
---

#### [5.4.5.2.10. __nv_atomic_load(), __nv_atomic_load_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-load-nv-atomic-load-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-load-nv-atomic-load-n "Permalink to this headline")

```cuda
__device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. | `__nv_atomic_load` stores `old` to where `ret` points to.
| `__nv_atomic_load_n` returns `old`.

The functions support the following data types:

- Any data type of size 1, 2, 4, 8 or 16 bytes.

`order` cannot be `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.
