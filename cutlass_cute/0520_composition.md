---
title: "Composition"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition"
---

## [Composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#composition)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#composition "Permalink to this headline")

Functional composition of `Layout`s is the core of CuTe and is used in just about every higher-level operation.

Starting again from the observation that `Layout`s are just functions from integers to integers, we can define functional composition that results in another `Layout`. First, an example.

```text
Functional composition, R := A o B
R(c) := (A o B)(c) := A(B(c))

Example
A = (6,2):(8,2)
B = (4,3):(3,1)

R( 0) = A(B( 0)) = A(B(0,0)) = A( 0) = A(0,0) =  0
R( 1) = A(B( 1)) = A(B(1,0)) = A( 3) = A(3,0) = 24
R( 2) = A(B( 2)) = A(B(2,0)) = A( 6) = A(0,1) =  2
R( 3) = A(B( 3)) = A(B(3,0)) = A( 9) = A(3,1) = 26
R( 4) = A(B( 4)) = A(B(0,1)) = A( 1) = A(1,0) =  8
R( 5) = A(B( 5)) = A(B(1,1)) = A( 4) = A(4,0) = 32
R( 6) = A(B( 6)) = A(B(2,1)) = A( 7) = A(1,1) = 10
R( 7) = A(B( 7)) = A(B(3,1)) = A(10) = A(4,1) = 34
R( 8) = A(B( 8)) = A(B(0,2)) = A( 2) = A(2,0) = 16
R( 9) = A(B( 9)) = A(B(1,2)) = A( 5) = A(5,0) = 40
R(10) = A(B(10)) = A(B(2,2)) = A( 8) = A(2,1) = 18
R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42
```

The absolutely amazing observation is that the function `R(c) = k` defined above can be written down as another `Layout`

```console
R = ((2,2),3):((24,2),8)
```

AND

```console
compatible(B, R)
```

That is, every coordinate of `B` can also be used as a coordinate of `R`. This is an expected property of functional composition because `B` defines the _domain_ of `R`.

You can find many examples and checked post-conditions in [the `composition` unit test](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/composition.cpp). The post-conditions are precisely as we just stated.

```cpp
// @post compatible(@a layout_b, @a result)
// @post for all i, 0 <= i < size(@a layout_b), @a result(i) == @a layout_a(@a layout_b(i)))
Layout composition(LayoutA const& layout_a, LayoutB const& layout_b)
```
