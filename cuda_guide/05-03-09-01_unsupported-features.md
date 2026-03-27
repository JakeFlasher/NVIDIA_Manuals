---
title: "5.3.9.1. Unsupported Features"
section: "5.3.9.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#unsupported-features"
---

### [5.3.9.1. Unsupported Features](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#unsupported-features)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#unsupported-features "Permalink to this headline")

- Run-Time Type Information (RTTI) and exceptions are not supported in device code:
  - `typeid` keyword
  - `dynamic_cast` keyword
  - `try/catch/throw` keywords
- `long double` is not supported in device code.
- Trigraphs are not supported on any platform. Digraphs are not supported on Windows.
- User-defined `operator new`, `operator new[]`, `operator delete`, or `operator delete[]` cannot be used to replace the corresponding built-ins provided by the compiler, and it is considered undefined behavior on both host and device.
