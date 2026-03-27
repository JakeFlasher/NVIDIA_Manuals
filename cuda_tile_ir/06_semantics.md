---
title: "6. Semantics"
section: "6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#semantics"
---

# [6. Semantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections#semantics)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#semantics "Permalink to this headline")

This section provides a written English presentation of the operational semantics of **Tile IR**. These semantics
are intended to provide an understanding of **Tile IR** for: 1) those interested in generating **Tile IR** as a
code generation target, or 2) those interested in reading **Tile IR** programs produced by others.

It does **not** attempt to formalize every possible behavior that might be admitted by an axiomatic formulation
or a small step operational semantics. For understanding an even more informal presentation of the language and
its core concepts see the [Programming Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#section-prog-model) section.

We first introduce the abstract machine state and language definitions before describing semantics of individual
kernels and programs.

We then discuss the semantics of broad classes of operations as well, for more detailed descriptions of individual
operations and their behavior see [Operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#section-operations) for a complete listing.
