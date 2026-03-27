---
title: "Tagged Iterators"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#tagged-iterators"
---

### [Tagged Iterators](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tagged-iterators)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tagged-iterators "Permalink to this headline")

Any random-access iterator can be used to construct a `Tensor`, but
users can also “tag” any iterator with a memory space –
e.g., to indicate this iterator is accessing global memory or shared memory.
This is done by calling `make_gmem_ptr(g)` or `make_gmem_ptr<T>(g)` to tag `g` as a global memory iterator,
and `make_smem_ptr(s)` or `make_smem_ptr<T>(s)` to tag `s` as a shared memory iterator.

Tagging memory makes it possible for CuTe’s `Tensor` algorithms
to use the fastest implementation for the specific kind(s) of memory.
When calling very specific operations with `Tensor`s, it also allows those
operators to verify the tags against what is expected.
For example, some kinds of optimized copy operations require
the source of the copy to be global memory
and the destination of the copy to be shared memory.
Tagging makes it possible for CuTe to dispatch
to those copy operations and/or verify against those copy operations.
