---
title: "Limitations of Dynamic Control Flow"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#limitations-of-dynamic-control-flow"
---

### [Limitations of Dynamic Control Flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#limitations-of-dynamic-control-flow)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#limitations-of-dynamic-control-flow "Permalink to this headline")

- Early-exit `break`, `continue`, `pass` or raising exception from
control flow body are not yet supported.
- Operations in the control flow body are traced only when tracing is active in
that region.
- Values originating in control flow body are not available outside the control
flow.
- Changing type of a variable in control flow body is not allowed.

**Example:**

```python
@cute.jit
def control_flow_negative_examples(predicate: cutlass.Boolean):
    n = 10

    # ❌ This loop is dynamic, early-exit isn't allowed.
    for i in range(n):
        if i == 5:
            break         # Early-exit

    if predicate:
        val = 10
        # ❌ return from control flow body is not allowed.
        return
        # ❌ Raising exception from control flow body is not allowed.
        raise ValueError("This is not allowed")
        # ❌ Using pass in control flow body is not allowed.
        pass

    # ❌ val is not available outside the dynamic if
    cute.printf("%d\\n", val)

    if predicate:
        # ❌ Changing type of a variable in control flow body is not allowed.
        n = 10.0
```
