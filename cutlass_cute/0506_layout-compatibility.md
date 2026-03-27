---
title: "Layout compatibility"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layout-compatibility"
---

### [Layout compatibility](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#layout-compatibility)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#layout-compatibility "Permalink to this headline")

We say that layout A is _compatible_ with layout B if the shape of A is compatible with the shape of B.
Shape A is compatible with shape B if

- the size of A is equal to the size of B and
- all coordinates within A are valid coordinates within B.

For example:

- Shape 24 is NOT compatible with Shape 32.
- Shape 24 is compatible with Shape (4,6).
- Shape (4,6) is compatible with Shape ((2,2),6).
- Shape ((2,2),6) is compatible with Shape ((2,2),(3,2)).
- Shape 24 is compatible with Shape ((2,2),(3,2)).
- Shape 24 is compatible with Shape ((2,3),4).
- Shape ((2,3),4) is NOT compatible with Shape ((2,2),(3,2)).
- Shape ((2,2),(3,2)) is NOT compatible with Shape ((2,3),4).
- Shape 24 is compatible with Shape (24).
- Shape (24) is NOT compatible with Shape 24.
- Shape (24) is NOT compatible with Shape (4,6).

That is, _compatible_ is a weak partial order on Shapes as it is reflexive, antisymmetric, and transitive.
