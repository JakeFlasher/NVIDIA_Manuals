---
title: "Software Pipelining"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html#software-pipelining"
---

### [Software Pipelining](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#software-pipelining)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#software-pipelining "Permalink to this headline")

Software pipelining is a technique used to optimize loops. Typically, this involves writing a prefetch loop and a main loop.

```python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    # prefetch loop
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i], ...)

    # main loop
    for i in range(bound):
        if i + prefetch_stages < bound:
            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

        use(buffer[i % total_stages])

    ...
```

This can be tedious to write and tune. CuTe DSL provides a loop attribute to ask the compiler to do this.

```python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
        # Compiler automatically handles the pipelining:
        # - Generates prefetch loop for initial stages
        # - In main loop, prefetches future data while using current data
        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
        use(buffer[i % total_stages])  # Uses data from previous iterations

    ...
```

Compiler will automatically generate the prefetch loop with *prefetch_stages* iterations and a corresponding main loop.

This feature is experimental and only supported on sm90 and above.
