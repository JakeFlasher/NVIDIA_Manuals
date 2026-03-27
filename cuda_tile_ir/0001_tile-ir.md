---
title: "Tile IR"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/index.html#tile-ir"
---

# [Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest#tile-ir)[](https://docs.nvidia.com/cuda/tile-ir/latest/#tile-ir "Permalink to this headline")

This document provides the comprehensive specification for **Tile IR**, a portable, low-level tile virtual machine and
instruction set that models the GPU as a tile-based processor. Unlike the traditional SIMT machine model, **Tile IR** enables
native programming of hardware in terms of tiles (multi-dimensional array fragments).

**Tile IR** offers a high-level yet explicit target for code generation that abstracts away generation-specific hardware details while exposing the massive
parallelism of the GPU for next-generation compute frameworks in a stable and versioned manner. This document is tailored for users building or debugging
systems that produce **Tile IR** bytecode, covering the **Tile IR** programming model, syntax, type system, semantics, binary bytecode format, and memory consistency model.
