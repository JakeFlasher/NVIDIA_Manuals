---
title: "When to Use Explicit Conversion?"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#when-to-use-explicit-conversion"
---

### [When to Use Explicit Conversion?](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#when-to-use-explicit-conversion)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#when-to-use-explicit-conversion "Permalink to this headline")

The DLPack protocol is a widely used protocol for interoperability between different frameworks.
However, there is some associated overhead. Based on our benchmark, it usually takes between 2 to 3
us per call to `from_dlpack`.

Explicit conversion allows for caching the converted CuTe tensors in order to avoid the overhead of
repeated calls to `from_dlpack`.

```python
x = torch.randn(30, 20, device="cpu")
if key not in cached_tensors:
    # Do the conversion only for cache misses
    cached_tensors[key] = cute.runtime.from_dlpack(x)
foo(cached_tensors[key])
```

Another use case for explicit conversion is to gain fine-grain control over which modes of a tensor
are considered dynamic from the perspective of the generated program.
