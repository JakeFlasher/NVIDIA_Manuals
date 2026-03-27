---
title: "10.1. Platform & Compatibility Guarantees"
section: "10.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#platform-compatibility-guarantees"
---

## [10.1. Platform & Compatibility Guarantees](https://docs.nvidia.com/cuda/tile-ir/latest/sections#platform-compatibility-guarantees)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#platform-compatibility-guarantees "Permalink to this headline")

**Bytecode Stability**
The **Tile IR** bytecode format ensures that programs can be interpreted and loaded by all conforming drivers (see [Binary Format](https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#section-bytecode)).

**Program Portability**
A program conforming to **Tile IR** vX.Y is syntactically portable to any platform that advertises support for vX.Y or newer.

Portability does not imply availability of target-specific features on all targets: if a program uses a feature that the selected hardware
target does not support, the compiler will either:

> - diagnose the incompatibility
> - or apply a lowering that preserves the semantics defined by the specification

**CUDA Compatibility**
**Tile IR** respects the CUDA platform’s forward and backward minor-version compatibility rules for toolchain and driver integration (see [CUDA Minor Version Compatibility Rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)).
