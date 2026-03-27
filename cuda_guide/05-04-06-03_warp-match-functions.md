---
title: "5.4.6.3. Warp Match Functions"
section: "5.4.6.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-match-functions"
---

### [5.4.6.3. Warp Match Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-match-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-match-functions "Permalink to this headline")

> **Hint**
>
> It is suggested to use the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/warp/warp_match_all.html) `cuda::device::warp_match_all()` function as a generalized and safer alternative to `__match_all_sync` function.

```cuda
unsigned __match_any_sync(unsigned mask, T value);
unsigned __match_all_sync(unsigned mask, T value, int *pred);
```

The warp match functions perform a broadcast-and-compare operation of a variable between non-exited threads within a [warp](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-warps-simt).

**`__match_any_sync`**

  Returns the mask of non-exited threads that have the same bitwise `value` in `mask`.

**`__match_all_sync`**

  Returns `mask` if all non-exited threads in `mask` have the same bitwise `value`; otherwise 0 is returned. Predicate `pred` is set to `true` if all non-exited threads in `mask` have the same bitwise `value`; otherwise the predicate is set to false.

`T` can be `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`.

The functions are subject to the [Warp __sync Intrinsic Constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-sync-intrinsic-constraints).

> **Warning**
>
> These intrinsics do not provide any memory ordering.
