---
title: "Other output formats"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html#other-output-formats"
---

#### [Other output formats](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#other-output-formats)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#other-output-formats "Permalink to this headline")

Some CuTe types have special printing functions that use a different output format.

The `cute::print_layout` function will display any rank-2 layout in a plain text table. This is excellent for visualizing the map from coordinates to indices.

The `cute::print_tensor` function will display any rank-1, rank-2, rank-3, or rank-4 tensor in a plain text multidimensional table. The values of the tensor are printed so you can verify the tile of data is what you expect after a copy, for example.

The `cute::print_latex` function will print LaTeX commands that you can use to build a nicely formatted and colored tables via `pdflatex`. This work for `Layout`, `TiledCopy`, and `TiledMMA`, which can be very useful to get a sense of layout patterns and partitioning patterns within CuTe.
