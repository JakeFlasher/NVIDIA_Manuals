---
title: "While Loops"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#while-loops"
---

## [While Loops](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#while-loops)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#while-loops "Permalink to this headline")

Standard Python `while` is supported.

- **Condition without annotation** → lowered to intermediate representation (IR).
- **Condition annotated with `cutlass.const_expr`** → evaluated at compile time.

**Example:**

```python
@cute.jit
def main(dynamic_var: cutlass.Int32):
    n = 0

    # ✅ This is Python while loop, evaluated at compile time.
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\\n")
        n += 1

    # ✅ This is dynamic while loop, emitted IR while loop.
    while dynamic_var == 10:
        cute.printf("Dynamic True\\n")
        n += 1

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    while cutlass.const_expr(n < dynamic_var):
        n += 1
```
