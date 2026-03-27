---
title: "7.11. PTX interoperability"
section: "7.11"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#ptx-interoperability"
---

## [7.11. PTX interoperability](https://docs.nvidia.com/cuda/tile-ir/latest/sections#ptx-interoperability)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#ptx-interoperability "Permalink to this headline")

The axioms and relations of the **Tile IR** memory model are intended to be a strict weakening of the PTX memory model.

The **Tile IR** memory model is designed to allow communication with PTX threads.
Release and acquire patterns in **Tile IR** will match up with acquire and release patterns in PTX to build PTX _causality_ and **Tile IR** _happens before_.

The same is true for data races, data races between accesses in a **Tile IR** program and a PTX
program will result in undefined behavior as if the data race were all in **Tile IR**.
