---
title: "Interface and specialization opportunities"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#interface-and-specialization-opportunities"
---

### [Interface and specialization opportunities](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#interface-and-specialization-opportunities)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#interface-and-specialization-opportunities "Permalink to this headline")

A `Tensor` encapsulates the data type, data location,
and possibly also the shape and stride of the tensor at compile time.
As a result, `copy` can and does dispatch,
based on the types of its arguments,
to use any of various synchronous or asynchronous hardware copy instructions.

The `copy` algorithm has two main overloads.
The first just takes the source `Tensor` and the destination `Tensor`.

```c++
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

The second takes those two parameters, plus a `Copy_Atom`.

```c++
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

The two-parameter `copy` overload picks a default implementation
based only on the types of the two `Tensor` parameters.
The `Copy_Atom` overload lets callers override that default
by specifying a nondefault `copy` implementation.
