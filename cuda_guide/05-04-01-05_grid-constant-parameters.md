---
title: "5.4.1.5. __grid_constant__ Parameters"
section: "5.4.1.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#grid-constant-parameters"
---

### [5.4.1.5. __grid_constant__ Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#grid-constant-parameters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#grid-constant-parameters "Permalink to this headline")

Annotating a `__global__` function parameter with `__grid_constant__` prevents the compiler from creating a per-thread copy of the parameter. Instead, all threads in the grid will access the parameter through a single address, which can improve performance.

The `__grid_constant__` parameter has the following properties:

- It has the lifetime of the kernel.
- It is private to a single kernel, meaning the object is not accessible to threads from other grids, including sub-grids.
- All threads in the kernel see the same address.
- It is read-only. Modifying a `__grid_constant__` object or any of its sub-objects, including `mutable` members, is undefined behavior.

Requirements:

- Kernel parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types.
- All function declarations must be consistent with any `__grid_constant__` parameters.
- Function template specializations must match the primary template declaration with respect to any `__grid_constant__` parameters.
- Function template instantiations must also match the primary template declaration with respect to any `__grid_constant__` parameters.

Examples:

```cuda
struct MyStruct {
    int         x;
    mutable int y;
};

__device__ void external_function(const MyStruct&);

__global__ void kernel(const __grid_constant__ MyStruct s) {
    // s.x++; // Compile error: tried to modify read-only memory
    // s.y++; // Undefined Behavior: tried to modify read-only memory

    // Compiler will NOT create a per-thread local copy of "s":
    external_function(s);
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/Goq9jrEeo).
