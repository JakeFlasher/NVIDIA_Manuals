---
title: "In a nutshell"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.html#in-a-nutshell"
---

## [In a nutshell](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#in-a-nutshell)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#in-a-nutshell "Permalink to this headline")

When using the `@jit` or `@kernel` decorators to define a JIT-compiled function, the arguments to the function are traced to determine the JIT function’s signature.
CuTe DSL provides a Pythonic way to write the arguments for JIT function as one normally would in Python, and the CuTe DSL will take care of the rest for you.

Specifically, CuTe DSL honors following when generating the JIT function’s arguments:

- JIT function arguments are assumed to be **dynamic arguments** by default.
- If an argument is explicitly type annotated with `cutlass.Constexpr`, it is treated as a **compile-time constant**.
- If type annotation is provided, CuTe DSL validates the argument type at compile time for **type safety**.
- CuTe DSL provides **runtime checkable protocols** (`JitArgument` and `DynamicExpression`) for generating JIT function arguments for customized types.

More details below for each of the above.
