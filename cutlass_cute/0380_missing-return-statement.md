---
title: "Missing return statement"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#missing-return-statement"
---

#### [Missing return statement](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#missing-return-statement)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#missing-return-statement "Permalink to this headline")

GCC 10 (but not 7.5, 9.4.0, or 11) has trouble deducing that a function with `auto` return type and all of its returns in an `if constexpr` … `else` statement must actually return.  As a result, GCC emits spurious “missing return statement” build warnings.  Such functions have one of two forms: `if constexpr` … `else` where `else` returns, and `if constexpr` … `else` where `else` is meant to fail at compile time.  Here is an example of the first form.

```c++
template<class T>
constexpr auto first_form(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
  else {
    return yet_another_function(t);
  }
}
```

In this form, the `if constexpr` … `else` sequence of branches covers all possibilities.  Here is an example of the second form.

```c++
template<class T>
constexpr auto second_form(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
  else {
    static_assert(sizeof(T) < 0, "This branch always fails");
  }
}
```

In this form, the `else` branch had a `static_assert` that was meant always to fail if the `else` branch were taken, such as `static_assert(sizeof(T) < 0)`.  (Note that we cannot use `static_assert(false)` here, because it will ALWAYS fail at compile time, even if the `else` branch is not taken.  C++23 fixes this behavior, but CUTLASS currently requires that its code be compatible with C++17.  As a result, CUTLASS includes a `dependent_false<T>` library function that you can use in place of the always-`false` test `sizeof(T) < 0`.)

One can suppress “missing return statement” warnings for both forms by invoking CUTLASS’ function-like macro `CUTE_GCC_UNREACHABLE`.  When building with GCC, this invokes the GCC-specific built-in function `__builtin_unreachable()`.  Actually calling this function is undefined behavior, so using this lets the programmer declare that the code path calling that function will never be taken.  (C++23 introduces the `std::unreachable()` function, which achieves the same goal.  Again, though, CUTLASS cannot currently use C++23 library functions.)  Here is an example of how to use `CUTE_GCC_UNREACHABLE`.

```c++
template<class T>
constexpr auto second_form(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
  else {
    static_assert(sizeof(T) < 0, "This branch always fails");
  }
  CUTE_GCC_UNREACHABLE;
}
```

This macro should only be used if it is needed to suppress spurious warnings.  Also, this function should not be used if the developer is not sure whether the code exhaustively tests all possibilities.  For example, some functions may look like this.

```c++
template<class T>
constexpr auto possibly_nonexhaustive(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }

  // NOTE lack of unadorned "else" here
}
```

This is a good opportunity to review the function.  If the branches are obviously meant to be exhaustive, you can add an `else` branch with a `static_assert` (see above for how to express this).  If you’re not sure, leave it alone and let the compiler issue warnings.
