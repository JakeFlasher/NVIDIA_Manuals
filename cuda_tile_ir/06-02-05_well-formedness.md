---
title: "6.2.5. Well-Formedness"
section: "6.2.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#well-formedness"
---

### [6.2.5. Well-Formedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections#well-formedness)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#well-formedness "Permalink to this headline")

A well-formed module is a module that satisfies the following properties:

- The module contains at least one item.
- Each item is uniquely named within the module.
- Each Tile Kernel and Tile Function has a body that is a sequence of statements in valid static-single-assignment (SSA) form.
- The program type checks according to the rules specified in [Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#section-types) and the operator type signatures are specified in [Operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#section-operations).

Program well-formedness is required as a pre-condition and post-condition of both optimizations and operational semantic rules.
