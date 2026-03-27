---
title: "Using a Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#using-a-layout"
---

### [Using a Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#using-a-layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#using-a-layout "Permalink to this headline")

The fundamental use of a `Layout` is to map between coordinate space(s) defined by the `Shape` and an index space defined by the `Stride`. For example, to print an arbitrary rank-2 layout in a 2-D table, we can write the function

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

which produces the following output for the above examples.

```console
> print2D(s2xs4)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
```

We can see static, dynamic, row-major, column-major, and hierarchical layouts printed here. The statement `layout(m,n)` provides the mapping of
the logical 2-D coordinate (m,n) to the 1-D index.

Interestingly, the `s2xh4` example isn’t row-major or column-major. Furthermore, it has three modes but is still interpreted as rank-2 and we’re using a 2-D coordinate. Specifically, `s2xh4` has a 2-D multi-mode in the second mode, but we’re still able to use a 1-D coordinate for that mode. More on this in the next section, but first we can generalize this another step. Let’s use a 1-D coordinate and treat all of the modes of each layout as a single multi-mode.  For instance, the following `print1D` function

```c++
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

produces the following output for the above examples.

```console
> print1D(s2xs4)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)
  0    1    2    3    4    5    6    7
> print1D(s2xh4)
  0    4    2    6    1    5    3    7
```

Any multi-mode of a layout, including the entire layout itself, can accept a 1-D coordinate. More on this in the following sections.

CuTe provides more printing utilities for visualizing Layouts. The `print_layout` function produces a formatted 2-D table of the Layout’s mapping.

```text
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

The `print_latex` function generates LaTeX that can be compiled with `pdflatex` into a color-coded vector graphics image of the same 2-D table.
