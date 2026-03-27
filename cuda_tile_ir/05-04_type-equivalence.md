---
title: "5.4. Type Equivalence"
section: "5.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-equivalence"
---

## [5.4. Type Equivalence](https://docs.nvidia.com/cuda/tile-ir/latest/sections#type-equivalence)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-equivalence "Permalink to this headline")

**Tile IR** does not provide means to name types. Equivalence of types
hence is a purely structural property: Two types are considered equal if
they are structurally identical.

Note that some types form a natural subtype relationship. Types with
dynamic shapes and strides like view cover the values that an identical
type with all dynamic shapes and strides substituted with static values
would cover. However, we consider these types distinct in **Tile IR**.
