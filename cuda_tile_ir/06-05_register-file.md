---
title: "6.5. Register File"
section: "6.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#register-file"
---

## [6.5. Register File](https://docs.nvidia.com/cuda/tile-ir/latest/sections#register-file)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#register-file "Permalink to this headline")

The register file, \(\(R\)\), maps named registers to values. Each assignment (i.e., SSA variable) in the tile
function’s body is assigned to a unique register which eventually holds the value of the operation’s result.
Registers are local to a tile block and are not visible to other tile blocks. As stated previously, the memory
representation of values in registers are not visible to the program and is not specified by the language semantics.

Values will only be fetched or persisted to global memory by memory operations (see [Memory](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-group-memory)).

The register file is indexed by the tile block’s coordinates, \(\(\vec{g}\)\), and the register’s name,
\(\(r\)\) and produces a value \(\(v\)\).

$$
\[S.R(\vec{g}, r) = v\]
$$
