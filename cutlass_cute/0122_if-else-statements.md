---
title: "If-Else Statements"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#if-else-statements"
---

## [If-Else Statements](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#if-else-statements)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#if-else-statements "Permalink to this headline")

Standard Python `if`/`elif`/`else` is supported.

- **Predicate without annotation** → lowered to intermediate representation (IR).
- **Predicate annotated with `cutlass.const_expr`** → evaluated at compile time.

**Example:**

```python
@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\\n")
    else:
        cute.printf("Const else\\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        cute.printf("Dynamic True\\n")
    else:
        cute.printf("Dynamic False\\n")

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\\n")
```
