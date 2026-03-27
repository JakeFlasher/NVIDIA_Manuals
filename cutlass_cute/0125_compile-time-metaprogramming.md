---
title: "Compile-Time Metaprogramming"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#compile-time-metaprogramming"
---

## [Compile-Time Metaprogramming](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#compile-time-metaprogramming)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#compile-time-metaprogramming "Permalink to this headline")

Mix compile-time constructs with normal CuTe DSL code to generate specialised
kernels without runtime overhead.  A compile-time flag can, for example, toggle
an optional **ReLU** epilogue:

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # main GEMM work
    ...
    if cutlass.const_expr(do_relu):    # compile-time guard
        # ReLU code is emitted only when do_relu is True
        ...
```

```text
gemm(..., False)   # ReLU is omitted from the generated |IR|
gemm(..., True)    # ReLU is included
```
