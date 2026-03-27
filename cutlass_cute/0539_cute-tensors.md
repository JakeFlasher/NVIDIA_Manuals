---
title: "CuTe Tensors"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#cute-tensors"
---

# [CuTe Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cute-tensors)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cute-tensors "Permalink to this headline")

This document describes `Tensor`, CuTe’s core container that deploys the `Layout` concepts previously described.

Fundamentally, a `Tensor` represents a multidimensional array. `Tensor`s abstracts away the details of how the array’s elements are organized and how the array’s elements are stored. This lets users write algorithms that access multidimensional arrays generically and potentially specialize algorithms on a `Tensor`s traits. For example, the rank of the `Tensor` can be dispatched against, the `Layout` of data can be inspected, and the type of data can be verified.

A `Tensor` is represented by two template parameters: `Engine` and `Layout`.
For a description of `Layout`, please refer to [the `Layout` section](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html).
The `Tensor` presents the same shape and access operators as the `Layout` and uses the result of the `Layout` computation to
offset and dereference a random-access iterator held by the `Engine`.
That is, the layout of the data is provided by `Layout` and the actual data is provided by the iterator. Such data can live in any kind of memory – global memory, shared memory, register memory – or can even be transformed or generated on the fly.
