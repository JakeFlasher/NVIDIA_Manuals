---
title: "1. Introduction"
section: "1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html#introduction"
---

# [1. Introduction](https://docs.nvidia.com/cuda/tile-ir/latest/sections#introduction)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#introduction "Permalink to this headline")

This document describes **Tile IR**, a portable, low-level tile virtual machine and instruction set.

Unlike PTX, which models the GPU as a data-parallel single instruction multiple thread (SIMT) processor, **Tile IR** models the GPU as a tile-based processor.
In **Tile IR**, each logical thread (tile block) computes over partial fragments (tiles) of multi-dimensional arrays (tensors).
