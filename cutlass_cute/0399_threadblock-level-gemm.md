---
title: "Threadblock-level GEMM"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#threadblock-level-gemm"
---

### [Threadblock-level GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#threadblock-level-gemm)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#threadblock-level-gemm "Permalink to this headline")

Each threadblock computes its portion of the output GEMM by iteratively loading tiles of input
matrices and computing an accumulated matrix product. At the threadblock level, data are loaded from
global memory. The blocking strategy in general is key to achieving efficiency. However, the programmer
must balance multiple conflicting goals. A
larger threadblock means fewer fetches from global memory, thereby ensuring that DRAM bandwidth
does not become a bottleneck.
However, large threadblock tiles may not match the dimensions of the problem well. If either the
GEMM _M_ or _N_ dimension is small, some threads within the threadblock may not perform meaningful
work, as the threadblock may be partially outside the bounds of the problem. If both _M_ and _N_
are small while _K_ is large, this scheme may launch relatively few threadblocks and fail to
make full use of all multiprocessors within the GPU. Strategies to optimize performance for this case,
as described in the section [Parallelized Reductions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#parallelized-reductions),
partition the GEMM K dimension across multiple threadblocks or multiple warps. These threadblocks
or warps compute matrix products in parallel; the products are then reduced to compute the result.

In CUTLASS, the dimensions of the threadblock tile are specified as `ThreadblockShape::{kM, kN, kK}`
and may be tuned to specialize the GEMM computation for the target processor and dimensions of
the GEMM problem.
