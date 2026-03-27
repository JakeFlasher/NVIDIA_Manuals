---
title: "CuTe"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#code_organization--cute"
---

## [CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cute)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cute "Permalink to this headline")

CuTe is a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data. CuTe provides `Layout` and `Tensor` objects that compactly packages the type, shape, memory space, and layout of data, while performing the complicated indexing for the user. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations. More documentation
for CuTe can be found in [`cute/`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html).
