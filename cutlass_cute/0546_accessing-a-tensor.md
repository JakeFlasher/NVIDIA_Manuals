---
title: "Accessing a Tensor"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#accessing-a-tensor"
---

## [Accessing a Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#accessing-a-tensor)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#accessing-a-tensor "Permalink to this headline")

Users can access the elements of a `Tensor` via `operator()` and `operator[]`,
which take `IntTuple`s of logical coordinates.

When users access a `Tensor`,
the `Tensor` uses its `Layout` to map the logical coordinate
to an offset that can be accessed by the iterator.
You can see this in `Tensor`’s implementation of `operator[]`.

```c++
template <class Coord>
decltype(auto) operator[](Coord const& coord) {
  return data()[layout()(coord)];
}
```

For example, we can read and write to `Tensor`s using natural coordinates, using the variadic `operator()`, or the container-like `operator[]`.

```c++
Tensor A = make_tensor<float>(Shape <Shape < _4,_5>,Int<13>>{},
                              Stride<Stride<_12,_1>,    _64>{});
float* b_ptr = ...;
Tensor B = make_tensor(b_ptr, make_shape(13, 20));

// Fill A via natural coordinates op[]
for (int m0 = 0; m0 < size<0,0>(A); ++m0)
  for (int m1 = 0; m1 < size<0,1>(A); ++m1)
    for (int n = 0; n < size<1>(A); ++n)
      A[make_coord(make_coord(m0,m1),n)] = n + 2 * m0;

// Transpose A into B using variadic op()
for (int m = 0; m < size<0>(A); ++m)
  for (int n = 0; n < size<1>(A); ++n)
    B(n,m) = A(m,n);

// Copy B to A as if they are arrays
for (int i = 0; i < A.size(); ++i)
  A[i] = B[i];
```
