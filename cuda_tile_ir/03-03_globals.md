---
title: "3.3. Globals"
section: "3.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/syntax.html#globals"
---

## [3.3. Globals](https://docs.nvidia.com/cuda/tile-ir/latest/sections#globals)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#globals "Permalink to this headline")

A global variable definition is a variable that is defined outside of a kernel.

```default
global_variable_definition ::= `global` <symbol_name> `:` <type> `=` <value>
```
