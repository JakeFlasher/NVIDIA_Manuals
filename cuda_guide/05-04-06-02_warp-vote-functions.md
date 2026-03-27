---
title: "5.4.6.2. Warp Vote Functions"
section: "5.4.6.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-vote-functions"
---

### [5.4.6.2. Warp Vote Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-vote-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-vote-functions "Permalink to this headline")

```cuda
int      __all_sync   (unsigned mask, int predicate);
int      __any_sync   (unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
```

The warp vote functions enable the threads of a given [warp](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-warps-simt) to perform a reduction-and-broadcast operation. These functions take an integer `predicate` as input from each non-exited thread in the warp and compare those values with zero. The results of the comparisons are then combined (reduced) across the [active threads](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#simt-architecture-notes) of the warp in one of the following ways, broadcasting a single return value to each participating thread:

**`__all_sync(unsigned mask, predicate)`:**

  Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if `predicate` evaluates to non-zero for all of them.

**`__any_sync(unsigned mask, predicate)`:**

  Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if `predicate` evaluates to non-zero for one or more of them.

**`__ballot_sync(unsigned mask, predicate)`:**

  Evaluates `predicate` for all non-exited threads in `mask` and returns an integer whose Nth bit is set if `predicate` evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. Otherwise, the Nth bit is zero.

The functions are subject to the [Warp __sync Intrinsic Constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-sync-intrinsic-constraints).

> **Warning**
>
> These intrinsics do not provide any memory ordering.
