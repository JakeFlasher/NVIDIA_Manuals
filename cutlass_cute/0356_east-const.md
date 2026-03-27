---
title: "East const"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#east-const"
---

#### [East const](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#east-const)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#east-const "Permalink to this headline")

CUTLASS uses the
[“East const”](http://slashslash.info/2018/02/a-foolish-consistency/)
convention.
That is, the `const` or `constexpr` keyword
goes after the type, not before.
The general rule is that `const` or `constexpr`
modifies the type to the left of it.
Here are some examples.

```c++
float constexpr compile_time_constant = 42.3f;

float const const_float = /* whatever */;
float const& reference_to_const_float = const_float;
float const* pointer_to_const_float = &const_float;
float const* const const_pointer_to_const_float = &const_float;

float nonconst_float;
float& reference_to_nonconst_float = nonconst_float;
float* pointer_to_nonconst_float = &nonconst_float;
float* const pointer_to_nonconst_float = &nonconst_float;
```

Contrast this with “West const” style, e.g.,

```c++
const float const_float = /* whatever */;
const float* pointer_to_const_float = &const_float;
```
