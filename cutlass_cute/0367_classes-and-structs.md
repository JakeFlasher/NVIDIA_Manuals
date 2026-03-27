---
title: "Classes and structs"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#classes-and-structs"
---

#### [Classes and structs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#classes-and-structs)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#classes-and-structs "Permalink to this headline")

Type names use `CamelCase`.
That is, words start with capital letters.
The remaining letters in the word are lower case,
and words are joined with no intervening underscores.
The only exception is when implementations are
a drop-in replacement for C++ Standard Library components.

Follow the
[C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-struct)
to decide whether to use `class` or `struct`.

- Use `class` when the object must maintain an invariant.
Data members related to the invariant should be `private`.
- Use `struct` when the class has no invariant to maintain,
and data members may vary arbitrarily with respect to each other.

Prefer nonmember functions and statelessness where possible.
Member functions imply invariants.
More invariants make code maintenance and testing harder.
