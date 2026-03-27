---
title: "7.2. Scopes"
section: "7.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#scopes"
---

## [7.2. Scopes](https://docs.nvidia.com/cuda/tile-ir/latest/sections#scopes)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#scopes "Permalink to this headline")

Memory operations in **Tile IR** may have a scope.
Operations without a scope are called `weak`.
All memory operations specify a scope or `weak`.
Any scope other than *weak* requires a memory ordering to be set.

| Scope | Description |
| --- | --- |
| `tile_block` | Tile block scope, for communication within a single tile block. |
| `device` | Device scope, for communication within the same GPU. |
| `sys` | System scope, for communication anywhere in the system. |

Weak operations cannot be used to communicate through memory between threads, or between fragments of the same tile block which are not ordered by token order.
The compiler may assume that tiles accessed with `weak` are not concurrently accessed by any other thread.

> **Note**
>
> Tile Block scope is needed when building communicating algorithms where there is communication within a single tile block.
> This is necessary when communicating memory through memory between operations which are not ordered by token order, or when storing to a tile with internal aliasing.
