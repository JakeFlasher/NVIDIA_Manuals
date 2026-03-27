---
title: "5.4.6.4. Warp Reduce Functions"
section: "5.4.6.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-reduce-functions"
---

### [5.4.6.4. Warp Reduce Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-reduce-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-reduce-functions "Permalink to this headline")

> **Hint**
>
> It is suggested to use the `CUB` [Warp-Wide “Collective” Primitives](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html#_CPPv4I0_iEN3cub10WarpReduceE)  to perform a Warp Reduction whenever possible for efficiency, safety, and portability reasons.

Supported by devices of compute capability 8.x or higher.

```cuda
T        __reduce_add_sync(unsigned mask, T value);
T        __reduce_min_sync(unsigned mask, T value);
T        __reduce_max_sync(unsigned mask, T value);

unsigned __reduce_and_sync(unsigned mask, unsigned value);
unsigned __reduce_or_sync (unsigned mask, unsigned value);
unsigned __reduce_xor_sync(unsigned mask, unsigned value);
```

The `__reduce_<op>_sync` intrinsics perform a reduction operation on the data provided in `value` after synchronizing all non-exited threads named in `mask`.

**`__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`**

  Returns the result of applying an arithmetic add, min, or max reduction operation on the values provided in `value` by each non-exited thread named in `mask`. `T` can be an `unsigned` or `signed` integer.

**`__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`**

  Returns the result of applying a bitwise AND, OR, or XOR reduction operation on the values provided in `value` by each non-exited thread named in `mask`.

The functions are subject to the [Warp __sync Intrinsic Constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-sync-intrinsic-constraints).

> **Warning**
>
> These intrinsics do not provide any memory ordering.
