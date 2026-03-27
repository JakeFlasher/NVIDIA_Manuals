---
title: "5.4.1.1. Execution Space Specifiers"
section: "5.4.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-space-specifiers"
---

### [5.4.1.1. Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#execution-space-specifiers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#execution-space-specifiers "Permalink to this headline")

The execution space specifiers `__host__`, `__device__`, and `__global__` indicate whether a function executes on the host or the device.

| Host | Device | Host | Device |  |
| --- | --- | --- | --- | --- |
| `__host__`, no specifier | ✅ | ❌ | ✅ | ❌ |
| `__device__` | ❌ | ✅ | ❌ | ✅ |
| `__global__` | ❌ | ✅ | ✅ | ✅ |
| `__host__ __device__` | ✅ | ✅ | ✅ | ✅ |

---

Constraints for `__global__` functions:

- Must return `void`.
- Cannot be a member of a `class`, `struct`, or `union`.
- Requires an execution configuration as described in [Kernel Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#execution-configuration).
- Does not support recursion.
- Refer to `__global__` [function parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#global-function-parameters) for additional restrictions.

Calls to a `__global__` function are asynchronous. They return to the host thread before the device completes execution.

---

Functions declared with `__host__ __device__` are compiled for both the host and the device. The `__CUDA_ARCH__` [macro](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-arch-macro) can be used to differentiate host and device code paths:

```cuda
__host__ __device__ void func() {
#if defined(__CUDA_ARCH__)
    // Device code path
#else
    // Host code path
#endif
}
```
