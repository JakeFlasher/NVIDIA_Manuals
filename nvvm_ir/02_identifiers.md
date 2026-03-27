---
title: "2. Identifiers"
section: "2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#identifiers"
---

# [2. Identifiers](https://docs.nvidia.com/cuda/nvvm-ir-spec#identifiers)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#identifiers "Permalink to this headline")

The name of a named global identifier must have the form:

`@[a-zA-Z$_][a-zA-Z$_0-9]*`

Note that it cannot contain the . character.

`[@%]llvm.nvvm.*` and `[@%]nvvm.*` are reserved words.
