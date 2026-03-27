---
title: "5.8.2. Device threads"
section: "5.8.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html#device-threads"
---

## [5.8.2. Device threads](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#device-threads)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#device-threads "Permalink to this headline")

Once a device thread makes progress:

- If it is part of a [Cooperative Grid](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73),
all device threads in its grid shall eventually make progress.
- Otherwise, all device threads in its [thread-block cluster](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters)
shall eventually make progress.

> [Note: Threads in other thread-block clusters are not guaranteed to eventually make progress. - end note.]
>
>
> [Note: This implies that all device threads within its thread block shall eventually make progress. - end note.]

Modify [[intro.progress.1]](https://eel.is/c++draft/intro.progress#1) as follows (modifications in **bold**):

The implementation may assume that any **host** thread will eventually do one of the following:

> 1. terminate,
> 2. invoke the function [std::this_thread::yield](https://en.cppreference.com/w/cpp/thread/yield) ([[thread.thread.this]](http://eel.is/c++draft/thread.thread.this)),
> 3. make a call to a library I/O function,
> 4. perform an access through a volatile glvalue,
> 5. perform a synchronization operation or an atomic operation, or
> 6. continue execution of a trivial infinite loop ([[stmt.iter.general]](http://eel.is/c++draft/stmt.iter.general)).

**The implementation may assume that any device thread will eventually do one of the following:**

> 1. **terminate**,
> 2. **make a call to a library I/O function**,
> 3. **perform an access through a volatile glvalue except if the designated object has automatic storage duration, or**
> 4. **perform a synchronization operation or an atomic read operation except if the designated object has automatic storage duration.**
>
>
> [Note: Some current limitations of device threads relative to host threads
> are implementation defects known to us, that we may fix over time.
> Examples include the undefined behavior that arises from device threads
> that eventually only perform volatile or atomic operations
> on automatic storage duration objects.
> However, other limitations of device threads relative to host threads
> are intentional choices.  They enable performance optimizations
> that would not be possible if device threads followed the C++ Standard strictly.
> For example, providing forward progress to programs
> that eventually only perform atomic writes or fences
> would degrade overall performance for little practical benefit. - end note.]

<details>
<summary>Examples of forward progress guarantee differences between host and device threads due to modifications to [[intro.progress.1]](https://eel.is/c++draft/intro.progress#1).</summary>

The following examples refer to the itemized sub-clauses of the implementation assumptions for host and device threads above
using “host.threads.<id>” and “device.threads.<id>”, respectively.

```cuda
1// Example: Execution.Model.Device.0
2// Outcome: grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration.
3__global__ void ex0(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
4    if (threadIdx.x == 0) {
5        while(atom.load(cuda::memory_order_relaxed) == 0);
6    } else if (threadIdx.x == 1) {
7        atom.store(1, cuda::memory_order_relaxed);
8    }
9}
```

```cuda
1// Example: Execution.Model.Device.1
2// Allowed outcome: No thread makes progress because device threads don't support host.threads.2.
3__global__ void ex1() {
4    while(true) cuda::std::this_thread::yield();
5}
```

```cuda
1// Example: Execution.Model.Device.2
2// Allowed outcome: No thread makes progress because device threads don't support host.threads.4
3// for objects with automatic storage duration (see exception in device.threads.3).
4__global__ void ex2() {
5    volatile bool True = true;
6    while(True);
7}
```

```cuda
1// Example: Execution.Model.Device.3
2// Allowed outcome: No thread makes progress because device threads don't support host.threads.5
3// for objects with automatic storage duration (see exception in device.threads.4).
4__global__ void ex3() {
5    cuda::atomic<bool, cuda::thread_scope_thread> True = true;
6    while(True.load());
7}
```

```cuda
1// Example: Execution.Model.Device.4
2// Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
3__global void ex4() {
4    while(true) { /* empty */ }
5}
```

</details>
