---
title: "5.4.9.2. __builtin_assume_aligned()"
section: "5.4.9.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#builtin-assume-aligned"
---

### [5.4.9.2. __builtin_assume_aligned()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#builtin-assume-aligned)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#builtin-assume-aligned "Permalink to this headline")

> **Hint**
>
> It is suggested to use the `cuda::std::assume_aligned()` function provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) ([C++ reference](https://en.cppreference.com/w/cpp/memory/assume_aligned.html)) as a portable and safer alternative to the built-in functions.

```cuda
void* __builtin_assume_aligned(const void* ptr, size_t align)
void* __builtin_assume_aligned(const void* ptr, size_t align, <integral type> offset)
```

The built-in functions enable the compiler to assume that the returned pointer is aligned to at least `align` bytes.

- The three parameter version enables the compiler to assume that `(char*) ptr - offset` is aligned to at least `align` bytes.

`align` must be a power of two and an integer literal.

Examples:

```cuda
void* res1 = __builtin_assume_aligned(ptr, 32);    // compiler can assume 'res1' is at least 32-byte aligned
void* res2 = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume 'res2 = (char*) ptr - 8' is at least 32-byte aligned
```
