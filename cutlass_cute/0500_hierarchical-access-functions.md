---
title: "Hierarchical access functions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#hierarchical-access-functions"
---

### [Hierarchical access functions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#hierarchical-access-functions)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#hierarchical-access-functions "Permalink to this headline")

`IntTuple`s and `Layout`s can be arbitrarily nested.
For convenience, we define versions of some of the above functions
that take a sequence of integers, instead of just one integer.
This makes it possible to access elements
inside of nested `IntTuple` or `Layout` more easily.
For example, we permit `get<I...>(x)`, where `I...` is a “C++ parameter pack” that denotes zero or more (integer) template arguments. These hierarchical access functions include the following.

- `get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...)`. Extract the `IN`th of the … of the `I1`st of the `I0`th element of `x`.
- `rank<I...>(x)  := rank(get<I...>(x))`. The rank of the `I...`th element of `x`.
- `depth<I...>(x) := depth(get<I...>(x))`. The depth of the `I...`th element of `x`.
- `shape<I...>(x)  := shape(get<I...>(x))`. The shape of the `I...`th element of `x`.
- `size<I...>(x)  := size(get<I...>(x))`. The size of the `I...`th element of `x`.

In the following examples, you’ll see use of `size<0>` and `size<1>` to determine loops bounds for the 0th and 1st mode of a layout or tensor.
