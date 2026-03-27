---
title: "Handling cases with grid dimensions that aren’t multiples of each other"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#handling-cases-with-grid-dimensions-that-aren-t-multiples-of-each-other"
---

### [Handling cases with grid dimensions that aren’t multiples of each other](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#handling-cases-with-grid-dimensions-that-aren-t-multiples-of-each-other)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#handling-cases-with-grid-dimensions-that-aren-t-multiples-of-each-other "Permalink to this headline")

Even though threadblock shapes M and N are typically multiples of one another, the grid
for a given problem may not have dimensions of the same ratio as that of the threadblock.
For example, a problem of size 132x132 using a threadblock of shape 64x32 will result
in a grid of 3x5 tiles. In this case, there is not an integer number of “true tiles”
per “macro tile.”

When this scenario arises, we simply pad the larger dimension of the grid such that
there are an integer number of “true tiles” per “macro tile.” Thus, the 3x5 grid in
the example above will be treated as a 3x6 grid. Row and column positions for each
tile are calculated as above. Any threadblocks that map to tiles that are outside the
problem range or upper/lower triangular portion (e.g., (2, 5)) will exit early from
this problem and may proceed to the next problem in the group.
