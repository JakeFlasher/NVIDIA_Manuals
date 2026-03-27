---
title: "Tensor Engines"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#tensor-engines"
---

## [Tensor Engines](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tensor-engines)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tensor-engines "Permalink to this headline")

The `Engine` concept is a wrapper for an iterator or array of data.
It uses a stripped-down interface of `std::array` to present the iterator.

```c++
using iterator     =  // The iterator type
using value_type   =  // The iterator value-type
using reference    =  // The iterator reference-type
iterator begin()      // The iterator
```

In general, users do not need to construct `Engine`s on their own. When a `Tensor` is constructed,
the appropriate engine – often `ArrayEngine<T,N>`, `ViewEngine<Iter>`, or
`ConstViewEngine<Iter>` – will be constructed.
