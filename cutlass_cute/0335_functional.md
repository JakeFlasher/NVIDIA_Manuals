---
title: "Functional"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#functional"
---

## [Functional](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#functional)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#functional "Permalink to this headline")

CUTLASS defines function objects corresponding to basic arithmetic operations modeled after C++ Standard Library’s `<functional>` header.

CUTLASS extends this by defining `multiply_add<T>` which computes `d = a * b + c`. The partial specialization `multiply_add<complex<T>>` computes complex-valued multiplication and addition using four real-valued multiply-add operations; these may correspond to native hardware instructions.

Example:

```c++
complex<float> a;
complex<float> b;
complex<float> c;
complex<float> d;

multiply_add<complex<float>> mad_op;

d = mad_op(a, b, c);    // four single-precision multiply-add instructions
```

CUTLASS defines partial specializations for type `Array<T, N>`, performing elementwise operations on each element. A further partial specialization for `Array<half_t, N>` targets may target native SIMD instructions for compute capability SM60 and beyond.

**Example:** Fused multiply-add of arrays of half-precision elements.

```c++
static int const kN = 8;

Array<half_t, kN> a;
Array<half_t, kN> b;
Array<half_t, kN> c;
Array<half_t, kN> d;

multiply_add<Array<half_t, kN>> mad_op;

d = mad_op(a, b, c);   // efficient multiply-add for Array of half-precision elements
```
