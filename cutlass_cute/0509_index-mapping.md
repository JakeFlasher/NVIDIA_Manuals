---
title: "Index Mapping"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#index-mapping"
---

#### [Index Mapping](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#index-mapping)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#index-mapping "Permalink to this headline")

The map from a natural coordinate to an index is performed by taking the inner product of the natural coordinate with the `Layout`’s `Stride`.

Take the layout `(3,(2,3)):(3,(12,1))`, for example. Then a natural coordinate `(i,(j,k))` will result in the index `i*3 + j*12 + k*1`. The indices this layout computes are shown in the 2-D table below where `i` is used as the row coordinate and `(j,k)` is used as the column coordinate.

```console
       0     1     2     3     4     5     <== 1-D col coord
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
```

The function `cute::crd2idx(c, shape, stride)` is responsible for the index mapping. It will take any coordinate within the shape, compute the equivalent natural coordinate for that shape (if it is not already), and compute the inner product with the strides.

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```
