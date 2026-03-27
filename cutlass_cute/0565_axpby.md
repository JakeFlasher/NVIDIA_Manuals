---
title: "axpby"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html#axpby"
---

## [axpby](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#axpby)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#axpby "Permalink to this headline")

The `axpby` algorithm lives in the header file
[`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp).
It assigns to $y$ the result of $\alpha x + \beta y$,
where $\alpha$ and $\beta$ are scalars and $x$ and $y$ are `Tensor`s.
The name stands for “Alpha times X Plus Beta times Y,”
and is a generalization of the original BLAS “AXPY” routine
(“Alpha times X Plus Y”).
