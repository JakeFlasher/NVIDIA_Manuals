---
title: "Suggestions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html#suggestions"
---

### [Suggestions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#suggestions)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#suggestions "Permalink to this headline")

For reliable and predictable results:

- Avoid dependent types in your code
- Implement explicit type conversion for dynamic values
- Clearly distinguish between static (compile-time) and dynamic (runtime) values
- Use type annotations as much as possible to help JIT compiler
to identify type to avoid ambiguity

```python
# Example demonstrating explicit typing
alpha = 1.0  # Explicitly defined as float using `1.0` instead of `1`
             #  or `float(1)`
beta = 2.0   # Explicitly defined as float
result = max(alpha, beta)  # Will correctly perform float comparison
```

****Debugging Capabilities****

  Debugging tools and facilities for the Python DSL are currently more limited in comparison to the C++
  API. For instance, we don’t support single-stepping through the JIT-compiled code. And lack of exception
  handling in JIT-compiled code makes it hard to debug in some cases.

****Integration with Frameworks****

  Integration with certain deep learning frameworks is in early development stages and may have
  limitations. For instance, converting frameworking tensor to cute.Tensor is known to have overhead
  with 2us~3us per tensor as we convert from general DLPack protocol which offers comptibility with
  all frameworks.

****Hashing DSL APIs and Objects****

  DSL APIs and Objects are sensitive to MLIR context, region or other contextual information which has no meaning cross
  different context. Any stateful design rely on `__hash__` likely misbehave with unexpected results. An example is
  `functools.lru_cache`, which combined with `@cute.jit`, it may cache MLIR object from one context and use in another one.
