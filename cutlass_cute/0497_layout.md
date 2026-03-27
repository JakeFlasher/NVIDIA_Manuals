---
title: "Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layout"
---

### [Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#layout "Permalink to this headline")

A `Layout` is a tuple of (`Shape`, `Stride`).
Semantically, it implements a mapping from
any coordinate within the Shape to an index via the Stride.
