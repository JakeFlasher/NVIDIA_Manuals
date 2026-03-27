---
title: "3.5. Types"
section: "3.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/syntax.html#types"
---

## [3.5. Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#types "Permalink to this headline")

A type in **Tile IR** is a fixed pre-defined set of types.

```default
element_type ::= `f32` | `f64` | `i8` | `i16` | `i32` | `i64` | `b8` | `b16` | `b32` | `b64`

type ::= `tile` `<` shape `x` element_type `>`

shape ::= `[` integer_literal (`x` integer_literal)* `]`
```
