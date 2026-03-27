---
title: "9.2.4. Subprogram"
section: "9.2.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#subprogram"
---

### [9.2.4. Subprogram](https://docs.nvidia.com/cuda/tile-ir/latest/sections#subprogram)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#subprogram "Permalink to this headline")

A subprogram represents a function within the source language. It specifies the scope, file, line number, name, and linkage name of the subprogram. Optionally, it can include the line number within the scope. Subprogram information is used for function level debugging, allowing developers to set breakpoints, step through code, and inspect stack frames within specific functions.

Fields:

- **File**: The source file containing the subprogram.
- **Line**: The line number where the subprogram starts.
- **Name**: The name of the subprogram.
- **Linkage Name**: The linkage name of the subprogram.
- **Compile Unit**: The compilation unit containing the subprogram.
- **Scope Line**: The line number where the scope of the subprogram starts. [Optional]

The following fields are set by the **Tile IR** compiler and listed for reference:

- **Subroutine Type**: Set to `() -> ()` meaning that all functions appear to have no arguments and no return value.
