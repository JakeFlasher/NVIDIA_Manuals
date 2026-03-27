---
title: "5.3.6.3. memcpy() and memset()"
section: "5.3.6.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#memcpy-and-memset"
---

### [5.3.6.3. memcpy() and memset()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#memcpy-and-memset)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#memcpy-and-memset "Permalink to this headline")

```cuda
__host__ __device__ void* memcpy(void* dest, const void* src, size_t size);
```

The function copies `size` bytes from the memory location pointed by `src` to the memory location pointed by `dest`.

```cuda
__host__ __device__ void* memset(void* ptr, int value, size_t size);
```

The function sets `size` bytes of memory block pointed by `ptr` to `value`, interpreted as an `unsigned char`.

> **Hint**
>
> It is suggested to use the `cuda::std::memcpy()` and `cuda::std::memset()` functions provided in the `<cuda/std/cstring>` [header](https://nvidia.github.io/cccl/libcudacxx/standard_api/c_library/cstring.html#libcudacxx-standard-api-cstring) as safer versions of `memcpy` and `memset`.
