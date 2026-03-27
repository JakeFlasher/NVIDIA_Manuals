---
title: "5.3.13.1. Three-way Comparison Operator"
section: "5.3.13.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#three-way-comparison-operator"
---

### [5.3.13.1. Three-way Comparison Operator](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#three-way-comparison-operator)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#three-way-comparison-operator "Permalink to this headline")

The three-way comparison operator (`<=>`) is supported in device code, but some uses implicitly rely on functionality from the C++ Standard Library, which is provided by the host implementation. Using those operators may require specifying the flag `--expt-relaxed-constexpr` to silence warnings, and the functionality requires the host implementation to satisfy the requirements of the device code.

Examples:

```cuda
#include <compare> // std::strong_ordering implementation

struct S {
    int x, y;

    auto operator<=>(const S&) const = default; // (a)

    __host__ __device__ bool operator<=>(int rhs) const { return false; } // (b)
};

__host__ __device__ bool host_device_function(S a, S b) {
    if (a <=> 1)  // CORRECT, calls a user-defined host-device overload (b)
        return true;
    return a < b; // CORRECT, call to an implicitly-declared function (a)
                  // Note: it requires a device-compatible std::strong_ordering
                  //       implementation provided in the header <compare>
                  //       and the flag --expt-relaxed-constexpr
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/qzs5arfx4).
