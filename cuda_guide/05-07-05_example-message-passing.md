---
title: "5.7.5. Example: Message Passing"
section: "5.7.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#example-message-passing"
---

## [5.7.5. Example: Message Passing](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#example-message-passing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#example-message-passing "Permalink to this headline")

The following example passes a message stored to the `x` variable by a
thread in block `0` to a thread in block `1` via the flag `f`:

| **Initially** |
| --- |
| ```cpp int x = 0, f = 0; ``` |
| **Thread 0 Block 0** |
| ```cpp x = 42; cuda::atomic_ref<int, cuda::thread_scope_device> flag(f); flag.store(1, memory_order_release); ``` |
| **Thread 0 Block 1** |
| ```cpp cuda::atomic_ref<int, cuda::thread_scope_device> flag(f); while(flag.load(memory_order_acquire) != 1); assert(x == 42); ``` |

In the following variation of the previous example, two threads
concurrently access the `f` object without synchronization, which
leads to a **data race**, and exhibits **undefined behavior**:

| **Initially** |
| --- |
| ```cpp int x = 0, f = 0; ``` |
| **Thread 0 Block 0** |
| ```cpp x = 42; cuda::atomic_ref<int, cuda::thread_scope_block> flag(f); flag.store(1, memory_order_release); // UB: data race ``` |
| **Thread 0 Block 1** |
| ```cpp cuda::atomic_ref<int, cuda::thread_scope_device> flag(f); while(flag.load(memory_order_acquire) != 1); // UB: data race assert(x == 42); ``` |

While the memory operations on `f` - the store and the loads - are
atomic, the scope of the store operation is “block scope”. Since the
store is performed by Thread 0 of Block 0, it only includes all other
threads of Block 0. However, the thread doing the loads is in Block 1,
i.e., it is not in a scope included by the store operation performed in
Block 0, causing the store and the load to not be “atomic”, and
introducing a data-race.

For more examples see the [PTX memory consistency model litmus
tests](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms).
