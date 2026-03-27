---
title: "5. Atomics Application Binary Interface"
section: "5"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#atomics-application-binary-interface"
---

# [5. Atomics Application Binary Interface](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#atomics-application-binary-interface)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#atomics-application-binary-interface "Permalink to this headline")

The mappings of programming languages’ atomic operations to the PTX ISA need to
be implemented in a consistent manner across all programming languages that may
concurrently access shared memory.
The mapping from C++11 atomics for the CUDA architecture are proven correct in
[A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://dl.acm.org/doi/10.1145/3297858.3304043).
The PTX ISA provides atomic memory operations and fences for acquire, release,
acquire-release, and relaxed C++ memory ordering semantics.

> **Note**
>
> The memory order parameter is monotonic, and so it is valid to strengthen any such parameter. For example, it is valid to
> strengthen `fence.sc.<scope>; ld.relaxed.<scope>;` to `fence.sc.<scope>; ld.acquire.<scope>;` for sequentially consistent
> loads. The same applies for all mappings below.

> **Note**
>
> Where there is a choice of PTX ABI ISA mapping for a given C, C++, or CUDA C++ API it is acceptable to pick either and to
> mix mappings within the same binary.

The PTX ABI for C++ sequentially consistent atomic operations is the following:

| C or C++ or CUDA C++ API | PTX ABI ISA mapping |
| --- | --- |
| `atomic_thread_fence(memory_order_seq_cst, thread_scope_<scope>)` | `fence.sc.<scope>;` |
| `atomic_load(memory_order_seq_cst, thread_scope_<scope>)` | \| `fence.sc.<scope>; ld.acquire.<scope>;` (recommended) OR \| `fence.sc.<scope>; ld.relaxed.<scope>; fence.acquire.<scope>;` |
| `atomic_store(memory_order_seq_cst, thread_scope_<scope>)` | `fence.sc.<scope>; st.relaxed.<scope>;` |
| `atomic_<rmw op>(memory_order_seq_cst, thread_scope_<scope>)` | \| `fence.sc.<scope>; atom.acquire.<scope>.<rmw op>;` (recommended) OR \| `fence.sc.<scope>; atom.relaxed.<scope>.<rmw op>; fence.acquire.<scope>;` |

The PTX ABI for C++ release atomic operations is the following:

| C or C++ or CUDA C++ API | PTX ABI ISA mapping |
| --- | --- |
| `atomic_thread_fence(memory_order_release, thread_scope_<scope>)` | `fence.release.<scope>;` |
| `atomic_store(memory_order_release, thread_scope_<scope>)` | \| `st.release.<scope>;` (recommended) OR \| `fence.release.<scope>; st.relaxed.<scope>;` |
| `atomic_<rmw op>(memory_order_release, thread_scope_<scope>)` | \| `atom.release.<scope>.<rmw op>;` (recommended) OR \| `fence.release.<scope>; atom.relaxed.<scope>.<rmw op>;` |

The PTX ABI for C++ acquire atomic operations is the following:

| C or C++ or CUDA C++ API | PTX ABI ISA mapping |
| --- | --- |
| `atomic_thread_fence(memory_order_acquire, thread_scope_<scope>)` | `fence.acquire.<scope>;` |
| `atomic_load(memory_order_acquire, thread_scope_<scope>)` | \| `ld.acquire.<scope>;` (recommended) OR \| `ld.relaxed.<scope>; fence.acquire.<scope>;` |
| `atomic_<rmw op>(memory_order_acquire, thread_scope_<scope>)` | \| `atom.acquire.<scope>.<rmw op>;` (recommended) OR \| `atom.relaxed.<scope>.<rmw op>; fence.acquire.<scope>;` |

The PTX ABI for C++ acquire-release atomic operations is the following:

| C or C++ or CUDA C++ API | PTX ABI ISA mapping |
| --- | --- |
| `atomic_thread_fence(memory_order_acq_rel, thread_scope_<scope>)` | `fence.acq_rel.<scope>;` |
| `atomic_<rmw op>(memory_order_acq_rel, thread_scope_<scope>)` | \| `atom.acq_rel.<scope>.<rmw op>;` (recommended) OR \| `fence.release.<scope>; atom.acquire.<scope>.<rmw op>;` OR \| `fence.release.<scope>; atom.relaxed.<scope>.<rmw op>; fence.acquire.<scope>;` |

The PTX ABI for C++ relaxed atomic operations is the following:

| C or C++ or CUDA C++ API | PTX ABI ISA mapping |
| --- | --- |
| `atomic_load(memory_order_relaxed, thread_scope_<scope>)` | `ld.relaxed.<scope>;` |
| `atomic_store(memory_order_relaxed, thread_scope_<scope>)` | `st.relaxed.<scope>;` |
| `atomic_<rmw op>(memory_order_relaxed, thread_scope_<scope>)` | `atom.relaxed.<scope>.<rmw op>;` |
