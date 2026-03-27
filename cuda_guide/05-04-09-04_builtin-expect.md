---
title: "5.4.9.4. __builtin_expect()"
section: "5.4.9.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#builtin-expect"
---

### [5.4.9.4. __builtin_expect()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#builtin-expect)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#builtin-expect "Permalink to this headline")

```cuda
long __builtin_expect(long input, long expected)
```

The built-in function tells the compiler that `input` is expected to equal `expected`, and returns the value of `input`. It is typically used to provide branch prediction information to the compiler.
It behaves like the C++20 `[[likely]]` and `[[unlikely]]` [attributes](https://en.cppreference.com/w/cpp/language/attributes/likely).

Example:

```cuda
// indicate to the compiler that likely "var == 0"
if (__builtin_expect(var, 0))
    doit();
```
