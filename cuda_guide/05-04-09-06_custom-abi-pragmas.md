---
title: "5.4.9.6. Custom ABI Pragmas"
section: "5.4.9.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#custom-abi-pragmas"
---

### [5.4.9.6. Custom ABI Pragmas](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#custom-abi-pragmas)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#custom-abi-pragmas "Permalink to this headline")

The `#pragma nv_abi` directive enables applications compiled in [separate compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation) mode to achieve performance similar to that of [whole program compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation) by preserving the number of registers used by a function.

The syntax for using this pragma is as follows, where `EXPR` refers to any integral constant expression:

```cuda
#pragma nv_abi preserve_n_data(EXPR) preserve_n_control(EXPR)
```

- The arguments that follow `#pragma nv_abi` are optional and may be provided in any order; however, at least one argument is required.
- The `preserve_n` arguments limit the number of registers preserved during a function call:
  - `preserve_n_data(EXPR)` limits the number of data registers.
  - `preserve_n_control(EXPR)` limits the number of control registers.

The `#pragma nv_abi` directive can be placed immediately before a device function declaration or definition.

```cuda
#pragma nv_abi preserve_n_data(16)
__device__ void dev_func();

#pragma nv_abi preserve_n_data(16) preserve_n_control(8)
__device__ int dev_func() {
    return 0;
}
```

Alternatively, it can be placed directly before an indirect function call within a C++ expression statement inside a device function. Note that while indirect function calls to free functions are supported, indirect calls to function references or class member functions are not supported.

```cuda
__device__ int dev_func1();

struct MyStruct {
    __device__ int member_func2();
};

__device__ void test() {
    auto* dev_func_ptr = &dev_func1; // type: int (*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v1 = dev_func_ptr();         // CORRECT, indirect call

    #pragma nv_abi preserve_n_control(8)
    int v2 = dev_func1();            // WRONG, direct call; the pragma has no effect
                                     // dev_func1 has type: int(void)

    auto& dev_func_ref = &dev_func1; // type: int (&)(void)
    #pragma nv_abi preserve_n_control(8)
    int v3 = dev_func_ref();         // WRONG, call to a reference
                                     // the pragma has no effect

    auto member_function_ptr = &MyStruct::member_func2; // type: int (MyStruct::*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v4 = member_function_ptr();  // WRONG, indirect call to member function
                                     // the pragma has no effect
}
```

When applied to a device function’s declaration or definition, the pragma modifies the custom ABI properties for any calls to that function. When placed at an indirect function call site, it affects the ABI properties only for that specific call. Note that the pragma only affects indirect function calls when placed at a call site; it has no effect on direct function calls.

```cuda
#pragma nv_abi preserve_n_control(8)
__device__ int dev_func3();

__device__ int dev_func4();

__device__ void test() {
    int v1 = dev_func3();            // CORRECT, the pragma affects the direct call

    auto* dev_func_ptr = &dev_func4; // type: int (*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v2 = dev_func_ptr();         // CORRECT, the pragma affects the indirect call

    int v3 = dev_func_ptr();         // WRONG, the pragma has no effect
}
```

Note that a program is ill-formed if the pragma arguments for a function declaration and its corresponding definition do not match.
