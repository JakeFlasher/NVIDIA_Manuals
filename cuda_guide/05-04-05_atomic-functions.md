---
title: "5.4.5. Atomic Functions"
section: "5.4.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions"
---

## [5.4.5. Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomic-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomic-functions "Permalink to this headline")

Atomic functions perform read-modify-write operations on shared data, making them appear to execute in a single step. Atomicity ensures that each operation either completes fully or not at all, providing all participating threads with a consistent view of the data.

CUDA provides atomic functions in four ways:

**Extended CUDA C++ atomic functions, [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) and [cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html).**

  - They are allowed in both host and device code.
  - They follow the [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) semantics.
  - They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.

**Standard C++ atomic functions, [cuda::std::atomic](https://en.cppreference.com/w/cpp/atomic/atomic.html) and [cuda::std::atomic_ref](https://en.cppreference.com/w/cpp/atomic/atomic_ref.html).**

  - They are allowed in both host and device code.
  - They follow the [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) semantics.
  - They do not allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.

**Compiler [built-in atomic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-atomic-functions), `__nv_atomic_<op>()`.**

  - They have been available since CUDA 12.8.
  - They are only allowed in device code.
  - They follow the [C++ standard atomic memory order](https://en.cppreference.com/w/cpp/atomic/memory_order.html) semantics.
  - They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.
  - They have the same memory ordering semantics as [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html).
  - They support a subset of the data types allowed by [cuda::std::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) and [cuda::std::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), except for 128-bit data types.

**[Legacy atomic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#legacy-atomic-functions), `atomic<Op>()`.**

  - They are only allowed in device code.
  - They only support `memory_order_relaxed` [C++ atomic memory semantics](https://en.cppreference.com/w/cpp/atomic/memory_order.html).
  - They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations as part of the function name.
  - Unlike [built-in atomic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-atomic-functions), legacy atomic functions only ensure atomicity and do not introduce synchronization points (fences).
  - They support a subset of the data types allowed by [built-in atomic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-atomic-functions). The atomic `add` operation supports additional data types.

> **Hint**
>
> Using the [Extended CUDA C++ atomic functions](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) provided by `libcu++` is recommended for efficiency, safety, and portability.
