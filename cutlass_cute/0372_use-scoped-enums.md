---
title: "Use scoped enums"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#use-scoped-enums"
---

#### [Use scoped enums](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#use-scoped-enums)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#use-scoped-enums "Permalink to this headline")

Use scoped enums (a C++11 feature) for enumerated types.
Use capital letters for the enumerated type name
and prefix `k` for enumerators like other constants.

```c++
enum class MatrixOperation {
  kNone,
  kTranspose,
  kConjugate,
  kHermitian
};
```
