---
title: "Coordinate Mapping"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#coordinate-mapping"
---

#### [Coordinate Mapping](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#coordinate-mapping)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#coordinate-mapping "Permalink to this headline")

The map from an input coordinate to a natural coordinate is the application of a colexicographical order (reading right to left, instead of “lexicographical,” which reads left to right) within the `Shape`.

Take the shape `(3,(2,3))`, for example. This shape has three coordinate sets: the 1-D coordinates, the 2-D coordinates, and the natural (h-D) coordinates.

| 1-D | 2-D | Natural |  | 1-D | 2-D | Natural |
| --- | --- | --- | --- | --- | --- | --- |
| `0` | `(0,0)` | `(0,(0,0))` |  | `9` | `(0,3)` | `(0,(1,1))` |
| `1` | `(1,0)` | `(1,(0,0))` |  | `10` | `(1,3)` | `(1,(1,1))` |
| `2` | `(2,0)` | `(2,(0,0))` |  | `11` | `(2,3)` | `(2,(1,1))` |
| `3` | `(0,1)` | `(0,(1,0))` |  | `12` | `(0,4)` | `(0,(0,2))` |
| `4` | `(1,1)` | `(1,(1,0))` |  | `13` | `(1,4)` | `(1,(0,2))` |
| `5` | `(2,1)` | `(2,(1,0))` |  | `14` | `(2,4)` | `(2,(0,2))` |
| `6` | `(0,2)` | `(0,(0,1))` |  | `15` | `(0,5)` | `(0,(1,2))` |
| `7` | `(1,2)` | `(1,(0,1))` |  | `16` | `(1,5)` | `(1,(1,2))` |
| `8` | `(2,2)` | `(2,(0,1))` |  | `17` | `(2,5)` | `(2,(1,2))` |

Each coordinate into the shape `(3,(2,3))` has two _equivalent_ coordinates and all equivalent coordinates map to the same natural coordinate. To emphasize again, because all of the above coordinates are valid inputs, a Layout with Shape `(3,(2,3))` can be used as if it is a 1-D array of 18 elements by using the 1-D coordinates, a 2-D matrix of 3x6 elements by using the 2-D coordinates, or a h-D tensor of 3x(2x3) elements by using the h-D (natural) coordinates.

The previous 1-D print demonstrates how CuTe identifies 1-D coordinates with a colexicographical ordering of 2-D coordinates.  Iterating from `i = 0` to `size(layout)` and indexing into our layout with the single integer coordinate `i`, traverses the 2-D coordinates in this “generalized-column-major” order, even if the layout maps coordinates to indices in a row-major or more complex fashion.

The function `cute::idx2crd(idx, shape)` is responsible for the coordinate mapping. It will take any coordinate within the shape and compute the equivalent natural coordinate for that shape.

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```
