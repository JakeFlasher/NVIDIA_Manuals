---
title: "print() vs cute.printf(): Meta-Stage vs Object-Stage Output"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#print-vs-cute-printf-meta-stage-vs-object-stage-output"
---

### [print() vs cute.printf(): Meta-Stage vs Object-Stage Output](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#print-vs-cute-printf-meta-stage-vs-object-stage-output)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#print-vs-cute-printf-meta-stage-vs-object-stage-output "Permalink to this headline")

CuTe DSL provides two ways to print values, each operating at a different stage:

- **Python’s** `print()` — executes during the **meta-stage** (compilation).
Use it to inspect what the compiler sees.
- `cute.printf()` — compiles into the kernel and executes at **runtime** on
the GPU.  Use it to observe actual tensor values during execution.

The following examples demonstrate how the same `result` variable appears
differently depending on when and how you print it.

**Example 1: Dynamic variables (both** `a` **and** `b` **are runtime values)**

```python
@cute.jit
def add_dynamicexpr(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_dynamicexpr(5.0)
```

```text
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

At meta-stage, `result` is a proxy—its value is unknown until the kernel runs.
At runtime, `cute.printf()` prints the actual GPU-computed value.

**Example 2: Compile-time constants (both** `a` **and** `b` **are Constexpr)**

```python
@cute.jit
def add_constexpr(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_constexpr(5.0)
```

```text
$> python myprogram.py
[meta-stage] result = 7.0
[object-stage] result = 7.000000
```

Both values are known at compile time, so Python evaluates `2.0 + 5.0 = 7.0`
during tracing.  The constant is baked into the compiled kernel.

**Example 3: Hybrid (** `a` **is dynamic,** `b` **is Constexpr)**

```python
@cute.jit
def add_hybrid(b: cutlass.Constexpr):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_hybrid(5.0)
```

```text
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

The constant `b = 5.0` is folded in, but since `a` is dynamic, the result
remains a proxy at meta-stage.  The GPU computes the final answer at runtime.
