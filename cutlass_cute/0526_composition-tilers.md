---
title: "Composition Tilers"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition-tilers"
---

## [Composition Tilers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#composition-tilers)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#composition-tilers "Permalink to this headline")

In summary, a `Tiler` is one of the following objects.

1. A `Layout`.
2. A tuple of `Tiler`s.
3. A `Shape`, which will be interpreted as a tiler of `Layout`s with stride-1.

Any of the above can be used as the second argument in `composition`. With (1), we think of the `composition` as between two functions from integers to integers, no matter the ranks of the layouts. With (2) and (3), the `composition` is performed on each pair of corresponding modes of `A` and `B`, until case (1) is found.

This allows composition to be applied by-mode to retrieve arbitrary sublayouts of specified modes of a tensor (“Give me the 3x5x8 subblock of this MxNxL tensor”) but also allows entire tiles of data to be reshaped and reordered as if they were 1-D vectors (“Reorder this 8x16 block of data into a 32x4 block using this weird order of elements”). We will see the by-mode cases appear often when we are tiling for threadblocks in examples that follow. We will see 1-D reshaping and reordering when we want to apply arbitrary partitioning patterns for threads and values in MMAs in examples that follow.
