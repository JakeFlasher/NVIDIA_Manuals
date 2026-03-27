---
title: "A generic copy implementation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#a-generic-copy-implementation"
---

### [A generic copy implementation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#a-generic-copy-implementation)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#a-generic-copy-implementation "Permalink to this headline")

A simple example of a generic `copy` implementation
for any two `Tensor`s looks like this.

```c++
template <class TA, class ALayout,
          class TB, class BLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<TA, ALayout> const& src,  // Any logical shape
     Tensor<TB, BLayout>      & dst)  // Any logical shape
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

This generic `copy` algorithm addresses both `Tensor`s
with 1-D logical coordinates, thus traversing both `Tensor`s
in a logical column-major order.
Some reasonable architecture-independent optimizations
would include the following.

1. If the two `Tensor`s have known memory spaces with optimized
access instructions (like `cp.async`), then dispatch to the
custom instruction.
2. The two `Tensor`s have static layouts and it can be proven
that element vectorization is valid – for example, four `ld.global.b32`s
can be combined into a single `ld.global.b128` – then vectorize the source
and destinations tensors.
3. If possible, validate that the copy instruction to be used is
appropriate for the source and destination tensors.

CuTe’s optimized copy implementations can do all of these.
