---
title: "Macros"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#macros"
---

#### [Macros](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#macros)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#macros "Permalink to this headline")

Only use macros when the preprocessor
is the only way to accomplish the task.
Do not use macros for literal constants.
Instead, if inside the body of a function,
use `constexpr` values,
and if at namespace scope, use
[`inline constexpr` variables](https://en.cppreference.com/w/cpp/language/inline)
(a C++17 feature).

“Namespace” macros by starting them with the module name, e.g., `CUTLASS_`.
Macros and ONLY MACROS use all capital letters with underscores between words.
For example:

```c++
#define CUTLASS_MACROS_USE_ALL_CAPS inline __host__ __device__
```

Header files such as
[cutlass/cutlass.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cutlass.h)
and
[cute/config.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cutlass.h)
offer macros for expressing compiler-dependent behavior.
These include

- replacements for `__device__` and/or `__host__`
annotations:
  - `CUTLASS_HOST_DEVICE` or `CUTE_HOST_DEVICE`
for functions that run on the host and the device,
  - `CUTLASS_DEVICE` or `CUTE_DEVICE`
for functions that run on the device only,
  - `CUTE_HOST`
for functions that run on the host only, and
  - `CUTE_HOST_RTC`
for functions that run on the host only,
but occur as unevaluated operands (of e.g., `decltype` or `sizeof`;
see C++ Standard, `[expr.context]` 1) in device code; and
- annotations to loop unrolling:
  - `CUTLASS_PRAGMA_UNROLL` or `CUTE_UNROLL`
for full unrolling of loops with constant trip counts, and
  - `CUTLASS_PRAGMA_NO_UNROLL` or `CUTE_NO_UNROLL` to prevent unrolling.
