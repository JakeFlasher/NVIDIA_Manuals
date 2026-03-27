---
title: "2. Programming Model"
section: "2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#programming-model"
---

# [2. Programming Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections#programming-model)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#programming-model "Permalink to this headline")

**Tile IR** extends CUDA’s low-level programming model with new abstractions that differ from what has
previously existed in CUDA C++ or PTX.

This section introduces the programming model of **Tile IR** and familiarizes the reader with
its core concepts and abstractions. We do this by working through a series of real programs, building up to a dynamic,
high-performance implementation of GEMM that makes use of the all major features of **Tile IR**.

**Tile IR** has an expressive tile-based programming model. We introduce users to the various ways to represent
tensor computation by progressively adapting the example programs to take better advantage of **Tile IR** features
which help simplify programs and enable the compiler to provide performance portability. We start by first introducing
concepts that readers may be familiar with from prior art.
