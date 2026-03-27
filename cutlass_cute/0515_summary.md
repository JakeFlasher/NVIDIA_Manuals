---
title: "Summary"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#summary"
---

## [Summary](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#summary)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#summary "Permalink to this headline")

- The `Shape` of a `Layout` defines its coordinate space(s).
  - Every `Layout` has a 1-D coordinate space.
This can be used to iterate over the coordinate spaces in a colexicographical order.
  - Every `Layout` has a R-D coordinate space,
where R is the rank of the layout.
The colexicographical enumeration of the R-D coordinates
correspond to the 1-D coordinates above.
  - Every `Layout` has an h-D (natural) coordinate space where h is “hierarchical.” These are ordered colexicographically and the enumeration of that order corresponds to the 1-D coordinates above. A natural coordinate is _congruent_ to the `Shape` so that each element of the coordinate has a corresponding element of the `Shape`.
- The `Stride` of a `Layout` maps coordinates to indices.
  - The inner product of the elements of the natural coordinate with the elements of the `Stride` produces the resulting index.

For each `Layout` there exists an integral `Shape` that is that compatible with that `Layout`. Namely, that integral shape is `size(layout)`. We can then observe that

> Layouts are functions from integers to integers.

If you’re familiar with the C++23 feature `mdspan`,
this is an important difference between
`mdspan` layout mappings and CuTe `Layout`s. In CuTe, `Layout` is a first class citizen, is natively hierarchical to naturally represent functions beyond row-major and column-major, and can similarly be indexed with a hierarchy of coordinates.
(`mdspan` layout mappings can represent hierarchical functions as well,
but this requires defining a custom layout.)
Input coordinates for an `mdspan` must have the same shape as the `mdspan`;
a multidimensional `mdspan` does not accept 1-D coordinates.
