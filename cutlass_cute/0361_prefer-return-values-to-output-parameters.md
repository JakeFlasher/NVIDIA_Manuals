---
title: "Prefer return values to output parameters"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#prefer-return-values-to-output-parameters"
---

##### [Prefer return values to output parameters](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#prefer-return-values-to-output-parameters)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#prefer-return-values-to-output-parameters "Permalink to this headline")

In general, avoid in-out mutable references to return a value.
If you need to return multiple values,
you can return them by `struct` or `tuple`,
rather than by output references.
This includes the special case of error reporting
by returning either a value or an error code.
Please see the next section for details.

```c++
// Instead of passing in-out mutable references ...
void not_preferred(float& input_and_output); // not preferred

// keep functions pure and return value types instead
float preferred(float input); // preferred
```
