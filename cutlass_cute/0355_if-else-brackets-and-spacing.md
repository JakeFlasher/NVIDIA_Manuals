---
title: "If-else brackets and spacing"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#if-else-brackets-and-spacing"
---

#### [If-else brackets and spacing](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#if-else-brackets-and-spacing)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#if-else-brackets-and-spacing "Permalink to this headline")

- Always use braces with conditionals such as `if`,
even if the body is a single line.
- Use a space after control flow keywords
such as `if`, `for`, and `while`.
- Use a space after the parenthesis closing a conditional
such as `if`, and the curly bracket opening a scope.
- Use a new line between the closing brace
of an `if` branch, and the `else` keyword.

```c++
if (condition) { // space after if, and between ) and {
  // ... code ...
} // newline after }
else {
  // ... other code ...
}

// space after keyword for
for (int k = 0; k < num_iters; ++k) {
  // ... still more code ...
}
```
