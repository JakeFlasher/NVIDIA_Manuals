---
title: "Formatting function declarations and definitions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#formatting-function-declarations-and-definitions"
---

#### [Formatting function declarations and definitions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#formatting-function-declarations-and-definitions)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#formatting-function-declarations-and-definitions "Permalink to this headline")

Short function headers can go on one line.

Do not insert a newline between the parenthesis
that closes the function’s parameters and
the curly bracket that opens the function’s body.

```c++
int short_name(int x, int y) {
  return x + y;
}
```

If the function name and its parameters are too long to fit on one line,
break the line immediately after the opening parenthesis
that starts the parameter list.  Then, double-indent the parameters
to distinguish them from the body of the function.

```c++
void indeed_my_fellowbeings_this_function_name_is_unusually_long(
    std::uint32_t foo, // parameters are double-indented
    std::uint32_t const* bar,
    TypeA a,
    TypeB b,
    TypeC c) { // the ) and { go on the same line still
  auto d = body_of_the_function(a, b, c); // body is single-indented
  // ... more code ...
}
```

For a constructor with a long parameter list,
break the line after the parentheses, just as with other functions.
Align the colon that starts the constructor’s initializer list
flush with the comma on the next line.

As with functions, double-indent the parameters
to distinguish them from the constructor body.
Here is an example.

```c++
class YesTheCommunityAgreesThatTheNameOfThisClassIsIndeedExtremelyLong {
public:
  CUTLASS_HOST_DEVICE
  YesTheCommunityAgreesThatTheNameOfThisClassIsIndeedExtremelyLong(
      int this_is_the_first_parameter_and_its_name_is_long,
      int this_is_the_second_parameter_and_its_name_is_also_long,
      int this_is_the_third_parameter_and_its_name_is_long_too)
  : x_(this_is_the_first_parameter_and_its_name_is_long)
  , y_(this_is_the_second_parameter_and_its_name_is_also_long)
  , z_(this_is_the_third_parameter_and_its_name_is_long_too) {
    // constructor body
    // more of the constructor body
  }

private:
  int x_ = 0;
  int y_ = 0;
  int z_ = 0;
};
```
