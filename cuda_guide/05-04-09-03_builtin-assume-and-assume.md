---
title: "5.4.9.3. __builtin_assume() and __assume()"
section: "5.4.9.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#builtin-assume-and-assume"
---

### [5.4.9.3. __builtin_assume() and __assume()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#builtin-assume-and-assume)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#builtin-assume-and-assume "Permalink to this headline")

```cuda
void __builtin_assume(bool predicate)
void __assume        (bool predicate) // only with Microsoft Compiler
```

The built-in function enables the compiler to assume that the boolean argument is true. If the argument is false at runtime, the behavior is undefined. Note that if the argument has side effects, the behavior is unspecified.

Example:

```cuda
__device__ bool is_greater_than_zero(int value) {
    return value > 0;
}

__device__ bool f(int value) {
    __builtin_assume(value > 0);
    return is_greater_than_zero(value); // returns true, without evaluating the condition
}
```
