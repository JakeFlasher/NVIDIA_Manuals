---
title: "Return multiple values by struct or tuple"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#return-multiple-values-by-struct-or-tuple"
---

##### [Return multiple values by struct or tuple](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#return-multiple-values-by-struct-or-tuple)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#return-multiple-values-by-struct-or-tuple "Permalink to this headline")

Sometimes a function needs to return multiple values.  In that case, consider the following, in decreasing order of preference.

1. Return a `struct`.  This lets you name the fields
(for more self-documenting code),
yet still permits use of structured binding.
2. Return a `tuple`.  If you need a tuple type
that works on device, use `cute::tuple`.
(Please note that `cute::tuple` does not work
for all the types that work in `std::tuple`.
CuTe’s documentation explains.)
3. Resort to “returning” multiple values by output references
only if performance requires it.

Here is an example of the struct approach for named values.
For a comparable example in the C++ Standard,
please see [`std::allocate_at_least`](https://en.cppreference.com/w/cpp/memory/allocate_at_least),
which returns `std::allocation_result`.

```c++
struct my_computation_result {
  float value = 0.0f;
  float relative_error = 0.0f;
  bool success = false;
};

my_computation_result my_computation(float tolerance);

void foo(float tolerance) {
  // Approach 1: Use structured binding.  The names
  // you choose on the left-hand side have nothing
  // to do with the struct, so it's up to you
  // to get the order right.  On the other hand,
  // this code works whether my_computation returns
  // a struct or a tuple.
  auto [val, rel_err, ok] = my_computation(tolerance);

  // Approach 2: Keep the struct and use its named fields.
  // This approach prevents errors like mixing the order of return types.
  // However, it only works for structs, not for tuples.

  auto result = my_computation(tolerance);
  if (not result.success) {
    // computation did not succeed
  }
  else if (result.relative_error > tolerance) {
    // successful but relative error too large
  }
  else {
    // successful and relative error is in bounds
  }
}
```
