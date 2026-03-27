---
title: "5.4.9.5. __builtin_unreachable()"
section: "5.4.9.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#builtin-unreachable"
---

### [5.4.9.5. __builtin_unreachable()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#builtin-unreachable)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#builtin-unreachable "Permalink to this headline")

```cuda
void __builtin_unreachable(void)
```

The built-in function tells the compiler that the control flow will never reach the point at which the function is called. If the control flow does reach this point at runtime, the program has undefined behavior.

This function is useful for avoiding code generation of unreachable branches and disabling compiler warnings for unreachable code.

Example:

```cuda
// indicates to the compiler that the default case label is never reached.
switch (in) {
    case 1:  return 4;
    case 2:  return 10;
    default: __builtin_unreachable();
}
```
