---
title: "Introduction to TMA instructions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#introduction-to-tma-instructions"
---

## [Introduction to TMA instructions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#introduction-to-tma-instructions)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#introduction-to-tma-instructions "Permalink to this headline")

The Tensor Memory Accelerator (TMA) is a set of instructions for copying possibly multidimensional arrays between global and shared memory.  TMA was introduced in the Hopper architecture.  A single TMA instruction can copy an entire tile of data all at once.  As a result, the hardware no longer needs to compute individual memory addresses and issue a separate copy instruction for each element of the tile.

To accomplish this, the TMA instruction is given a _TMA descriptor_, which is a packed representation of a multidimensional tensor in global memory with 1, 2, 3, 4, or 5 dimensions. The TMA descriptor holds

- the base pointer of the tensor;
- the data type of the tensor’s elements (e.g., `int`, `float`, `double`, or `half`);
- the size of each dimension;
- the stride within each dimension; and
- other flags representing the smem box size, smem swizzling patterns, and out-of-bounds access behavior.

This descriptor must be created on the host before kernel execution.
It is shared between all thread blocks that will be issuing TMA instructions.
Once inside the kernel, the TMA is executed with the following parameters:

- pointer to the TMA descriptor;
- pointer to the SMEM; and
- coordinates into the GMEM tensor represented within the TMA descriptor.

For example, the interface for TMA-store with 3-D coordinates looks like this.

```cpp
struct SM90_TMA_STORE_3D {
  CUTE_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
    // ... invoke CUDA PTX instruction ...
  }
};
```

We observe that the TMA instruction does not directly consume pointers to global memory. Indeed, the global memory pointer is contained in the descriptor, is considered constant, and is NOT a separate parameter to the TMA instruction. Instead, the TMA consumes TMA coordinates into the TMA’s view of global memory that is defined in the TMA descriptor.

That means that an ordinary CuTe Tensor that stores a GMEM pointer and computes offsets and new GMEM pointers is useless to the TMA.

What do we do?
