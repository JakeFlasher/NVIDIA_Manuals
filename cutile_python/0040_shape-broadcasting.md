---
title: "Shape Broadcasting"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#shape-broadcasting"
---

## [Shape Broadcasting](https://docs.nvidia.com/cuda/cutile-python#shape-broadcasting)[](https://docs.nvidia.com/cuda/cutile-python/#shape-broadcasting "Permalink to this headline")

_Shape broadcasting_ allows [tiles](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) with different shapes to be combined in arithmetic operations.
When performing operations between [tiles](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) of different shapes, the smaller [tile](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) is automatically
extended to match the shape of the larger one, following these rules:

- [Tiles](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) are aligned by their trailing dimensions.
- If the corresponding dimensions have the same size or one of them is 1, they are compatible.
- If one [tile](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) has fewer dimensions, its shape is padded with 1s on the left.

Broadcasting follows the same semantics as [NumPy](https://numpy.org/doc/stable/), which makes code more concise and readable
while maintaining computational efficiency.
