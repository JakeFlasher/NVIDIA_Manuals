---
title: "Aside: Integer-module strides"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#aside-integer-module-strides"
---

#### [Aside: Integer-module strides](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#aside-integer-module-strides)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#aside-integer-module-strides "Permalink to this headline")

A group of objects that support addition between elements and product between elements and integers is called an integer-module.

Formally, an integer-module is an abelian group `(M,+)` equipped with `Z*M -> M`, where `Z` are the integers. That is, an integer-module `M` is
a group that supports inner products with the integers.
The integers are an integer-module.
Rank-R tuples of integers are an integer-module.

In principle, layout strides may be any integer-module.
