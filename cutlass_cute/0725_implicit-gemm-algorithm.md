---
title: "Implicit GEMM Algorithm"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#implicit-gemm-algorithm"
---

# [Implicit GEMM Algorithm](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#implicit-gemm-algorithm)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#implicit-gemm-algorithm "Permalink to this headline")

2-D convolution may be mapped to matrix multiply
by first forming a _convolution matrix_ containing elements of the activations tensor,
then multiplying this by a matrix formed from the filters tensor.
The earliest form of this algorithm constructs the convolution matrix explicitly via an operation
conventionally referred to as `im2col`. The resulting matrix replicates each activation element by a factor
equal to the filter size, consuming additional storage capacity and memory bandwidth.

The _implicit GEMM_ algorithm is a variation on the blocked, hierarchical GEMM computation in CUDA.
Instead of constructing the convolution matrix explicitly,
it forms tiles of the convolution matrix on the fly
as data are loaded from global memory into Shared Memory
by carefully updating pointers and predicates.
Once the convolution matrix is formed in Shared Memory,
the existing warp-level GEMM components accumulate the result of
convolution and update the output tensor.

This section describes the structure of an efficient Implicit GEMM Convolution CUDA kernel
for Turing Tensor Cores.
