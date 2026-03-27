---
title: "2.2.1. Implementing GEMM"
section: "2.2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#implementing-gemm"
---

### [2.2.1. Implementing GEMM](https://docs.nvidia.com/cuda/tile-ir/latest/sections#implementing-gemm)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#implementing-gemm "Permalink to this headline")

Now that we understand the basics concepts of the **Tile IR** programming model, we will introduce how to compute
a 2-d GEMM for a single block, and then generalize it step by step to a full GEMM by utilizing the
tile grid and introducing control flow and manual tiling.

This progression demonstrates how to compose basic tile operations into a complex, high-performance parallel algorithm.
