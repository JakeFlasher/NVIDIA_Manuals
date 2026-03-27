---
title: "Prefer using the actual return type to auto, if you know the type"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#prefer-using-the-actual-return-type-to-auto-if-you-know-the-type"
---

##### [Prefer using the actual return type to auto, if you know the type](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#prefer-using-the-actual-return-type-to-auto-if-you-know-the-type)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#prefer-using-the-actual-return-type-to-auto-if-you-know-the-type "Permalink to this headline")

C++ lets you use `auto` to deduce the type returned from a function.

- If you know the actual type, prefer using the type instead of `auto`.
- Use [Constructor Type Argument Deduction](https://en.cppreference.com/w/cpp/language/class_template_argument_deduction)
(CTAD) if you know that a function returns some type
(e.g., `Tensor`), but don’t know the type’s template arguments.
- Use `auto` in structured bindings (where you have to use it anyway).  This also makes your code agnostic of whether the return type is a `struct`, `tuple`, `pair`, or other tuple-like type.
- Be careful using `auto` with types that provide expression templates.

Contrast this with “Almost Always Auto” (AAA) style.
We deliberately choose not to follow AAA style,
for the following reasons.

- Using the actual type when we know it can help prevent common loss-of-precision errors in mixed-precision computations, an important use case for CUTLASS.
- CTAD gives us much of the brevity of AAA, with more clarity.
- Using the actual type instead of `auto` can prevent common dangling errors with expression templates.
