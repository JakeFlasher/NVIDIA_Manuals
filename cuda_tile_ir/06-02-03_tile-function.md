---
title: "6.2.3. Tile Function"
section: "6.2.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#tile-function"
---

### [6.2.3. Tile Function](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-function)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-function "Permalink to this headline")

A tile function consists of a name, a list of formal parameters, a return type, and a body.
A tile function’s body contains a single threaded tile program (referred to as _tile block_) parameterized
by formal parameters.

A tile function has \(\(N\)\) formal parameters  and produces *M* return values. The type of the parameters
can be one of the valid types described in [Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#section-types).

> **Note**
>
> Currently defining non-kernel tile functions is disabled with support planned for a future release.
