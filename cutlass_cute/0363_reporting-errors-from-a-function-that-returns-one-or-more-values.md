---
title: "Reporting errors from a function that returns one or more values"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#reporting-errors-from-a-function-that-returns-one-or-more-values"
---

##### [Reporting errors from a function that returns one or more values](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#reporting-errors-from-a-function-that-returns-one-or-more-values)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#reporting-errors-from-a-function-that-returns-one-or-more-values "Permalink to this headline")

We may want to return one or more values
from a function that could fail
or otherwise report errors.
That is, the function either

- returns one or more valid values, or
- does not return any values and reports an error,

but NOT BOTH.  We contrast this with cases
when it’s meaningful to report both a result
and whether the result is satisfactory.
For example, when solving
a system of nonlinear equations iteratively,
users may want the approximate computed solution,
even if the iteration did not succeed
by converging to the desired tolerance
in the desired number of steps.
(Users may want to invest more steps,
or use the current approximation
to jump-start a different algorithm.)

We’re talking here about the “either valid value(s),
or error, but not both” case.
For this case, C++ offers a few options.

1. Return the value(s), or throw an exception on error
2. `std::expected` (requiring C++23) or something like it
3. `std::optional` (for a Boolean error state)
or something like it
4. `std::variant` (a C++17 fall-back for `std::expected`)
or something like it
5. C-style interface: return an error code,
and “return” the values as output parameters

We usually cannot or do not want to
throw exceptions on device.
Some code projects forbid exceptions entirely
(on host or device)
and tell the compiler to disable them.
If we exclude a C-style interface (the last option)
as not idiomatic C++, then for host-only code,
`std::expected`, `std::optional`, and `std::variant`
all work.
For code that needs to build and run on device,
we can fall back to libcu++ equivalents
in the `cuda::std::` namespace, when they exist.
Otherwise, we must resort to returning a struct or tuple
with the value and the error information,
and ask users not to use the value on error.
This is acceptable if the value can be constructed
cheaply with a reasonable default.
