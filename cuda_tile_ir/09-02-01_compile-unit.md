---
title: "9.2.1. Compile Unit"
section: "9.2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#compile-unit"
---

### [9.2.1. Compile Unit](https://docs.nvidia.com/cuda/tile-ir/latest/sections#compile-unit)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#compile-unit "Permalink to this headline")

A compile unit represents the root scope of all objects declared within a specific compilation unit. It specifies the source file associated with the compilation unit and encompasses all the other debug information elements such as files, lexical blocks, and subprograms. This scope is fundamental for organizing and correlating debug information with the original source code, facilitating efficient debugging.

Fields:

- **File**: The source file associated with the compilation unit.

The following fields are set by the **Tile IR** compiler and listed for reference:

- **Is Optimized?**: Set to `false` if compiler option `--opt-level` is `0` else `true`.
- **Emission Kind**: Set based on compiler options: `--line-info` to generate line tables only or `--device-debug` alias `-g` for generating full debug info.
