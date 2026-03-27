---
title: "5.4.6.1. Warp Active Mask"
section: "5.4.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-active-mask"
---

### [5.4.6.1. Warp Active Mask](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-active-mask)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-active-mask "Permalink to this headline")

```cuda
unsigned __activemask();
```

The function returns a 32-bit integer mask representing all currently active threads in the calling warp. The Nth bit is set if the Nth lane in the warp is active when `__activemask()` is called. [Inactive threads](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#simt-architecture-notes) are represented by 0 bits in the returned mask. Threads that have exited the program are always marked as inactive.

> **Warning**
>
> `__activemask()` cannot be used to determine which warp lanes execute a given branch. This function is intended for opportunistic warp-level programming and only provides an instantaneous snapshot of the active threads within a warp.
>
>
> ```cuda
> // Check whether at least one thread's predicate evaluates to true
> if (pred) {
>     // Invalid: the value of 'at_least_one' is non-deterministic
>     // and could vary between executions.
>     at_least_one = __activemask() > 0;
> }
> ```

Note that threads convergent at an `__activemask()` call are not guaranteed to remain convergent at subsequent instructions unless those instructions are warp synchronizing intrinsics (`__sync`).

For example, the compiler could reorder instructions, and the set of active threads might not be preserved:

```cuda
unsigned mask      = __activemask();              // Assume mask == 0xFFFFFFFF (all bits set, all threads active)
int      predicate = threadIdx.x % 2 == 0;        // 1 for even threads, 0 for odd threads
int      result    = __any_sync(mask, predicate); // Active threads might not be preserved
```
