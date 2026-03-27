---
title: "5.4.6.6. Warp __sync Intrinsic Constraints"
section: "5.4.6.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints"
---

### [5.4.6.6. Warp __sync Intrinsic Constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#warp-sync-intrinsic-constraints)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#warp-sync-intrinsic-constraints "Permalink to this headline")

All warp `__sync` intrinsics, such as:

- `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`
- `__match_any_sync`, `__match_all_sync`
- `__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`, `__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`
- `__syncwarp`

use the `mask` parameter to indicate which warp threads participate in the call. This parameter ensures proper convergence before the hardware executes the intrinsic.

| Each bit in the `mask` corresponds to a thread’s lane ID (`threadIdx.x % warpSize`). The intrinsic waits until all non-exited warp threads specified in the `mask` reach the call.
| The following constraints must be met for correct execution:

- Each calling thread must have its corresponding bit set in the `mask`.
- Each non-calling thread must have its corresponding bit set to zero in the `mask`. Exited threads are ignored.
- All non-exited threads specified in the `mask` must execute the intrinsic with the same `mask` value.
- Warp threads may call the intrinsic concurrently with different `mask` values, provided the masks are disjoint. Such condition is valid even in divergent control flow.

The behavior of warp `__sync` functions is invalid, such as kernel hang, or undefined if:

- A calling thread is not specified in the `mask`.
- A non-exited thread specified in the `mask` fails to either eventually exit or call the intrinsic at the same program point with the same `mask` value.
- In conditional code, all conditions must evaluate identically across all non-exited threads specified in the `mask`.

> **Note**
>
> The intrinsics achieve the best efficiency when all warp threads participate in the call, namely when the `mask` is set to `0xFFFFFFFF`.

Examples of valid warp intrinsics usage:

```cuda
__global__ void valid_examples() {
    if (threadIdx.x < 4) {        // threads 0, 1, 2, 3 are active
        __all_sync(0b1111, pred); // CORRECT, threads 0, 1, 2, 3 participate in the call
    }

    if (threadIdx.x == 0)
        return; // exit
    // CORRECT, all non-exited threads participate in the call
    __all_sync(0xFFFFFFFF, pred);
}
```

Disjoint `mask` examples:

```cuda
__global__ void example_syncwarp_with_mask(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        unsigned mask = threadIdx.x < 16 ? 0xFFFF : 0xFFFF0000; // CORRECT
        __syncwarp(mask);
        if (threadIdx.x == 0 || threadIdx.x == 16)
            output_data[threadIdx.x] = shared_data[threadIdx.x + 1];
    }
}
```

```cuda
__global__ void example_syncwarp_with_mask_branches(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        if (threadIdx.x < 16) {
            unsigned mask = 0xFFFF; // CORRECT
            __syncwarp(mask);
            output_data[threadIdx.x] = shared_data[15 - threadIdx.x];
        }
        else {
            unsigned mask = 0xFFFF0000; // CORRECT
            __syncwarp(mask);
            output_data[threadIdx.x] = shared_data[31 - threadIdx.x];
        }
    }
}
```

Examples of invalid warp intrinsics usage:

```cuda
if (threadIdx.x < 4) {           // threads 0, 1, 2, 3 are active
    __all_sync(0b0000011, pred); // WRONG, threads 2, 3 are active but not set in mask
    __all_sync(0b1111111, pred); // WRONG, threads 4, 5, 6 are not active but set in mask
}

// WRONG, participating threads have a different and overlapping mask
__all_sync(threadIdx.x == 0 ? 1 : 0xFFFFFFFF, pred);
```
