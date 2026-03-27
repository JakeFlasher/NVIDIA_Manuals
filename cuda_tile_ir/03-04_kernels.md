---
title: "3.4. Kernels"
section: "3.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/syntax.html#kernels"
---

## [3.4. Kernels](https://docs.nvidia.com/cuda/tile-ir/latest/sections#kernels)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#kernels "Permalink to this headline")

A kernel definition is a function that is defined inside a **Tile IR** module.

```default
ssa_name := `%` identifier

function_signature ::= <function_parameter>*

function_parameter ::= <ssa_name> `:` <type>

<kernel_definition> ::= `func` @kernel_name `(` <function_signature> `)` `->` <function_type> {
    <kernel_body>
}
```

A kernel definition’s body is a sequence of operations.

```default
kernel_body ::= <operation>*

operation ::= (ssa_name `,`?)* `=` <operation_name> <ssa_name>* attribute=attribute_value : type ...
```
