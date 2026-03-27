---
title: "CuTe Layout Algebra"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#cute-layout-algebra"
---

# [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#cute-layout-algebra)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#cute-layout-algebra "Permalink to this headline")

CuTe provides an “algebra of `Layout`s” to support combining layouts in different ways.  This algebra includes operations such as

- `Layout` functional composition,
- a notion of `Layout` “product” to reproduce one layout according to another, and
- a notion of `Layout` “divide” to split one layout according to another.

Common utilities for building complicated layouts from simpler ones depend on the `Layout` product. Common utilities for partitioning layouts (of data, for example) across other layouts (of threads, for example) depend on the `Layout` divide. All of these utilities rely on the functional composition of `Layout`s.

In this section, we’ll build up the tools of the `Layout` algebra and explain some of these core operations in detail.
