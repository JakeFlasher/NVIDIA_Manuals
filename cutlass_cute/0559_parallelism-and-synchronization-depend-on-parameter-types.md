---
title: "Parallelism and synchronization depend on parameter types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#parallelism-and-synchronization-depend-on-parameter-types"
---

### [Parallelism and synchronization depend on parameter types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#parallelism-and-synchronization-depend-on-parameter-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#parallelism-and-synchronization-depend-on-parameter-types "Permalink to this headline")

Either the default implementation or
the implementation selected by a `Copy_Atom` overload
may use none or all available parallelism,
and may have a variety of synchronization semantics.
The behavior depends on `copy`’s parameter types.
Users are expected to figure this out based on their knowledge
of the architecture on which they are running.
(Developers often write a custom optimized kernel
for each GPU architecture.)

The `copy` algorithm may be sequential per thread,
or it may be parallel across some collection of threads
(e.g., a block or cluster).

If `copy` is parallel,
then the collection of participating threads
may need synchronization before any thread in the collection
may assume that the copy operation has completed.
For example, if the participating threads form a thread block,
then users must invoke `__syncthreads()`
or the Cooperative Groups equivalent
before they may use the results of `copy`.

The `copy` algorithm may use asynchronous copy instructions,
such as `cp.async`, or its C++ interface `memcpy_async`.
In that case, users will need to perform
the additional synchronization appropriate to that underlying implementation
before they may use the results of the `copy` algorithm.
[The CuTe GEMM tutorial example](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)
shows one such synchronization method.
More optimized GEMM implementations use pipelining techniques
to overlap asynchronous `copy` operations with other useful work.
