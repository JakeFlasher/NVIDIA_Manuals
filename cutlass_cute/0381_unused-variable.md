---
title: "Unused variable"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#unused-variable"
---

#### [Unused variable](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#unused-variable)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#unused-variable "Permalink to this headline")

Some compilers may emit spurious unused warnings for some variable declarations, where the variable was only being used inside a `decltype` in an `if constexpr` test. Marking the variables as `[[maybe_unused]]` (a standard C++17 attribute) suppresses these warnings.  Again, please only do this if you’re sure that the code is right.
