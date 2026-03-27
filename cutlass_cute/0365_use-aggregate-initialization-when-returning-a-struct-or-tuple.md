---
title: "Use aggregate initialization when returning a struct or tuple"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#use-aggregate-initialization-when-returning-a-struct-or-tuple"
---

##### [Use aggregate initialization when returning a struct or tuple](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#use-aggregate-initialization-when-returning-a-struct-or-tuple)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#use-aggregate-initialization-when-returning-a-struct-or-tuple "Permalink to this headline")

Use aggregate initialization when returning a struct or tuple.
This avoids duplication of the return type name.

```c++
struct foo_result {
  float value = 0.0f;
  float error = 0.0f;
  bool success = false;
};

foo_result foo(std::span<const float> input) {
  // ... code  ...

  // Prefer this.  We know what type the function returns.
  return {val, err, ok}; // prefer this

  // Naming foo_result again here is unnecessary.
  // return foo_result{val, err, ok};
}
```

However, note that this won’t work if the function returns `auto`.
The general rule is to avoid code duplication.

```c++
auto foo(std::span<const float> input) {
  // ... code  ...

  if constexpr (some_condition) {
    return foo_result{val, err, ok};
  }
  else {
    return bar_result{val, err, ok};
  }
}
```
