---
title: "5.4.5.2.9. __nv_atomic_compare_exchange(), __nv_atomic_compare_exchange_n()"
section: "5.4.5.2.9"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#nv-atomic-compare-exchange-nv-atomic-compare-exchange-n"
---

#### [5.4.5.2.9. __nv_atomic_compare_exchange(), __nv_atomic_compare_exchange_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-compare-exchange-nv-atomic-compare-exchange-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-compare-exchange-nv-atomic-compare-exchange-n "Permalink to this headline")

```cuda
__device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);

__device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Compare `old` with the value where `expected` points to.
3. If they are equal, the return value is `true` and `desired` is stored to where `address` points to. Otherwise, it returns `false` and `old` is stored to where `expected` points to.

The parameter `weak` is ignored and it picks the stronger memory order between `success_order` and `failure_order` to execute the compare-and-exchange operation.

The functions support the following data types:

- Any data type of size of 2, 4, 8 or 16 bytes.
- The 16-byte data type is supported on devices with compute capability 9.x and higher.
