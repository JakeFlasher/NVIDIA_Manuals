---
title: "Access the dumped contents programmatically"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#access-the-dumped-contents-programmatically"
---

### [Access the dumped contents programmatically](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#access-the-dumped-contents-programmatically)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#access-the-dumped-contents-programmatically "Permalink to this headline")

For compiled kernels, the generated PTX/CUBIN/IR can be accessed programmatically as well through following attributes:

- `__ptx__`: The generated PTX code of the compiled kernel.
- `__cubin__`: The generated CUBIN data of the compiled kernel.
- `__mlir__`: The generated IR code of the compiled kernel.

```python
compiled_foo = cute.compile(foo, ...)
print(f"PTX: {compiled_foo.__ptx__}")
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```
