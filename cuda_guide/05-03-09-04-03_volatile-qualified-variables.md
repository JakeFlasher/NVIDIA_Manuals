---
title: "5.3.9.4.3. volatile-qualified Variables"
section: "5.3.9.4.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#volatile-qualified-variables"
---

#### [5.3.9.4.3. volatile-qualified Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#volatile-qualified-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#volatile-qualified-variables "Permalink to this headline")

> **Note**
>
> The `volatile` keyword is supported to maintain compatibility with ISO C++. However, few, if any, of its [remaining non-deprecated uses](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop) apply to GPUs.

Reading and writing to `volatile`-qualified objects are not atomic and are compiled into one or more [volatile instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation) that do not guarantee:

- ordering of memory operations, or
- that the number of memory operations performed by the hardware matches the number of PTX instructions.

CUDA C++ `volatile` is NOT suitable for:

- **Inter-Thread Synchronization**: Use atomic operations via [cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html), or [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions) instead.

Atomic memory operations provide inter-thread synchronization guarantees and deliver better performance than `volatile` operations.
However, CUDA C++ `volatile` operations do not provide any inter-thread synchronization guarantees and are therefore not suitable for this purpose.
The following example shows how to pass a message between two threads using atomic operations.

**cuda::atomic_ref**
| ```cuda #include <cuda/atomic>  __global__ void kernel(int* flag, int* data) {     cuda::atomic_ref<int, cuda::thread_scope_device> atomic_ref{*flag};     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(atomic_ref.load(cuda::memory_order_acquire) == 0)             ;         if (*data != 42)             __trap(); // Errors if wrong data read     }     else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         atomic_ref.store(1, cuda::memory_order_release);     } } ``` |
| --- |

**cuda::atomic**
| ```cuda #include <cuda/atomic>  __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(flag->load(cuda::memory_order_acquire) == 0)             ;         if (*data != 42)             __trap(); // Errors if wrong data read     }     else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         flag->store(1, cuda::memory_order_release);     } } ``` |
| --- |

**Atomic Functions (atomicAdd and atomicExch)**
| ```cuda __global__ void kernel(int* flag, int* data) {     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(atomicAdd(flag, 0) == 0)             ;                // Load with Relaxed Read-Modify-Write         __threadfence();     // SequentiallyConsistent fence         if (*data != 42)             __trap();        // Errors if wrong data read     } else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         __threadfence();     // SequentiallyConsistent fence         atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write     } } ``` |
| --- |
- **Memory Mapped IO** (MMIO): Use [PTX MMIO operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation) via inline PTX instead.

PTX MMIO operations strictly preserve the number of memory accesses performed.
However, CUDA C++ `volatile` operations do not preserve the number of memory accesses performed and may perform more or fewer accesses than requested in an undetermined way. This makes them unsuitable for MMIO.
The following example shows how to read from and write to a register using PTX MMIO operations.

```cuda
__global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
    // Write to MMIO register:
    int value = 13;
    asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
        :
        : "l"(mmio_reg0), "r"(value) : "memory");
    // Read MMIO register:
    asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
        : "=r"(value)
        : "l"(mmio_reg1) : "memory");
    if (value != 42)
        __trap(); // Errors if wrong data read
}
```
