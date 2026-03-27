---
title: "Vector Layouts"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#vector-layouts"
---

### [Vector Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#vector-layouts)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#vector-layouts "Permalink to this headline")

We define a vector as any `Layout` with `rank == 1`.
For example, the layout `8:1` can be interpreted as an 8-element vector whose indices are contiguous.

```console
Layout:  8:1
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

Similarly,
the layout `8:2` can be interpreted as an 8-element vector where the indices of the elements are strided by `2`.

```console
Layout:  8:2
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  8 10 12 14
```

By the above rank-1 definition, we _also_ interpret layout `((4,2)):((2,1))` as a vector, since its shape is rank-1. The inner shape looks like a 4x2 row-major matrix, but the extra pair of parenthesis suggest we can interpret those two modes as a 1-D 8-element vector. The strides tell us that the first `4` elements are strided by `2` and then there are `2` of those first elements strided by `1`.

```console
Layout:  ((4,2)):((2,1))
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  1  3  5  7
```

We can see the second set of `4` elements are duplicates of the first `4` with an extra stride of `1`.

Consider the layout `((4,2)):((1,4))`. Again, it’s `4` elements strided by `1` and then `2` of those first elements strided by `4`.

```console
Layout:  ((4,2)):((1,4))
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

As a function from integers to integers, it’s identical to `8:1`. It’s the identity function.
