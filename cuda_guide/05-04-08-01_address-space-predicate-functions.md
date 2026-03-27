---
title: "5.4.8.1. Address Space Predicate Functions"
section: "5.4.8.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#address-space-predicate-functions"
---

### [5.4.8.1. Address Space Predicate Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#address-space-predicate-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#address-space-predicate-functions "Permalink to this headline")

Address space predicate functions are used to determine the address space of a pointer.

> **Hint**
>
> It is suggested to use the `cuda::device::is_address_from()` and `cuda::device::is_object_from()` functions provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory/is_address_from.html) as a portable and safer alternative to Address Space Predicate intrinsic functions.

```cuda
__device__ unsigned __isGlobal      (const void* ptr);
__device__ unsigned __isShared      (const void* ptr);
__device__ unsigned __isConstant    (const void* ptr);
__device__ unsigned __isGridConstant(const void* ptr);
__device__ unsigned __isLocal       (const void* ptr);
```

The functions return `1` if `ptr` contains the generic address of an object in the specified address space, `0` otherwise. Their behavior is unspecified if the argument is a `NULL` pointer.

- `__isGlobal()`: global memory space.
- `__isShared()`: shared memory space.
- `__isConstant()`: constant memory space.
- `__isGridConstant()`: kernel parameter annotated with `__grid_constant__`.
- `__isLocal()`: local memory space.
