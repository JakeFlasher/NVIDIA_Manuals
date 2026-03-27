---
title: "Alignment of reference and pointer types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#alignment-of-reference-and-pointer-types"
---

#### [Alignment of reference and pointer types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#alignment-of-reference-and-pointer-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#alignment-of-reference-and-pointer-types "Permalink to this headline")

For reference and pointer types,
align the `&` resp. `*` flush against the type
that it modifies.  This is called “left alignment.”

For example, do this:

```c++
int const& var;
int const* var;
```

and not this.

```c++
int const &var;
int const *var;
```
