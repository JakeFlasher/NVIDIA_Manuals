---
title: "C is not a subset of C++"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#c-is-not-a-subset-of-c"
---

#### [C is not a subset of C++](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#c-is-not-a-subset-of-c)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#c-is-not-a-subset-of-c "Permalink to this headline")

C is not a subset of C++.
Some valid C is not valid C++, and some valid “C-looking” C++ is not valid C.
See e.g., the informative C++ Standard Committee (WG21) document
[P2735R0](https://isocpp.org/files/papers/P2735R0.pdf),
which explains ways in which the same code has different behavior in C vs. C++.
In some cases, code that compiles in both C and C++,
and is correct in C, has undefined behavior (can crash or worse) in C++.
The “type.punning” section of P2735R0 specifically relates to unions.
