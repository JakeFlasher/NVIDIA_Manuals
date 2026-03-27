---
title: "10.4. Execution & Numerical Guarantees"
section: "10.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#execution-numerical-guarantees"
---

## [10.4. Execution & Numerical Guarantees](https://docs.nvidia.com/cuda/tile-ir/latest/sections#execution-numerical-guarantees)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#execution-numerical-guarantees "Permalink to this headline")

**Execution Determinism**
For a fixed toolchain, configuration, and hardware target, compilation and execution are deterministic within a single tile-block thread.

- **Version Changes**: Using a different toolchain version may produce a different program and thus different results; this is expected behavior, this is expected and not non-determinism.

**Numerical Stability**
**Tile IR** does not guarantee bit‑identical numerical results across different toolchain versions, configurations, or targets, except where explicitly documented.

- **Scope**: Stability guarantees are scoped to specific versions and targets.
- **Updates**: Changes are not retroactive; compiling/executing with an earlier toolchain retains the guarantees published for that version.

**Floating-point Semantics**
Floating‑point operations follow applicable IEEE semantics for the order in which they are actually evaluated.

- **Transformations**: Compiler transformations (e.g., reordering) can change numeric results across versions.
- **Precision**: Operations like MMA may have weaker or no guarantees of bit-identical numerical results unless explicitly documented.
