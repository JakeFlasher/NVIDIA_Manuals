---
title: "Basis elements"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#basis-elements"
---

#### [Basis elements](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#basis-elements)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#basis-elements "Permalink to this headline")

CuTe’s basis elements live in the header file `cute/numeric/arithmetic_tuple.hpp`.
To make it easy to create `ArithmeticTuple`s that can be used as strides, CuTe defines normalized basis elements using the `E` type alias. “Normalized” means that the scaling factor of the basis element is the compile-time integer 1.

| C++ object | Description | String representation |
| --- | --- | --- |
| `E<>{}` | `1` | `1` |
| `E<0>{}` | `(1,0,...)` | `1@0` |
| `E<1>{}` | `(0,1,0,...)` | `1@1` |
| `E<0,0>{}` | `((1,0,...),0,...)` | `1@0@0` |
| `E<0,1>{}` | `((0,1,0,...),0,...)` | `1@1@0` |
| `E<1,0>{}` | `(0,(1,0,...),0,...)` | `1@0@1` |
| `E<1,1>{}` | `(0,(0,1,0,...),0,...)` | `1@1@1` |

The “description” column in the above table
interprets each basis element as an infinite tuple of integers,
where all the tuple’s entries not specified by the element’s type are zero.
We count tuple entries from left to right, starting with zero.
For example, `E<1>{}` has a 1 in position 1: `(0,1,0,...)`.
`E<3>{}` has a 1 in position 3: `(0,0,0,1,0,...)`.

Basis elements can be _nested_.
For instance, in the above table, `E<0,1>{}` means that
in position 0 there is a `E<1>{}`: `((0,1,0,...),0,...)`. Similarly,
`1@1@0` means that `1` is lifted to position 1 to create `1@1`: `(0,1,0,...)`
which is then lifted again to position 0.

Basis elements can be _scaled_.
That is, they can be multiplied by an integer _scaling factor_.
For example, in `5*E<1>{}`, the scaling factor is `5`.
`5*E<1>{}` prints as `5@1` and means `(0,5,0,...)`.
The scaling factor commutes through any nesting.
For instance, `5*E<0,1>{}` prints as `5@1@0`
and means `((0,5,0,...),0,...)`.

Basis elements can also be added together,
as long as their hierarchical structures are compatible.
For example, `3*E<0>{} + 4*E<1>{}` results in `(3,4,0,...)`.
Intuitively, “compatible” means that
the nested structure of the two basis elements
matches well enough to add the two elements together.
