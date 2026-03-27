---
title: "Zero Compile and JIT Executor"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html#zero-compile-and-jit-executor"
---

## [Zero Compile and JIT Executor](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#zero-compile-and-jit-executor)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#zero-compile-and-jit-executor "Permalink to this headline")

Zero Compile is a feature that enables explicit kernel compilation on demand through `cute.compile`.
When `cute.compile` is called, it compiles the kernel and returns a JIT Executor instance.
This JIT Executor instance can be cached and reused directly for subsequent executions without compiling the kernel again.

The JIT Executor is a component that independently executes compiled code.
It can be created either through `cute.compile` or implicit compilation.
The JIT Executor instance behaves like a callable object to execute the compiled code.
Each JIT Executor instance maintains a single compiled host function.

It encompasses all necessary execution components:

- Host function pointer and its MLIR execution engine
- CUDA modules (optional)
- Argument specifications defining how Python arguments are converted to C ABI-compatible types. Note that arguments with the `cutlass.Constexpr` hint are excluded from argument specifications since they are evaluated at compile time rather than runtime.

For example, in the following code, `print_result` is a `cutlass.Constexpr` value that is **NOT** evaluated at runtime:

```python
import cutlass.cute as cute

@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b

jit_executor = cute.compile(add, 1, 2, True)

jit_executor(1, 2) # output: ``Result: 3``
```

The JIT Executor ensures all components are properly initialized and loaded after compilation.

For example, all CUDA modules are loaded (via `cuModuleLoad`) and kernel function pointers are extracted (via `cuModuleGetFunction`).

When calling a JIT Executor instance, it:

- Parses Python runtime arguments and converts them to C ABI-compatible types according to argument specifications
- Invokes the host function with the converted arguments
