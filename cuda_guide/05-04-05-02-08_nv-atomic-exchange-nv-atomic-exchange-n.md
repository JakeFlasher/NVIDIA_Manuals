---
title: "5.4.5.2.8. __nv_atomic_exchange(), __nv_atomic_exchange_n()"
section: "5.4.5.2.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-exchange-nv-atomic-exchange-n"
---

#### [5.4.5.2.8. __nv_atomic_exchange(), __nv_atomic_exchange_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-exchange-nv-atomic-exchange-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-exchange-nv-atomic-exchange-n "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. | `__nv_atomic_exchange_n` stores `val` to where `address` points to.
| `__nv_atomic_exchange` stores `old` to where `ret` points to and stores the value located at the address `val` to where `address` points to.

- `__nv_atomic_exchange_n` returns the `old` value.
- `__nv_atomic_exchange` has no return value.

The functions support the following data types:

- Any data type of size of 4, 8 or 16 bytes.
- The 16-byte data type is supported on devices of compute capability 9.x and higher.
