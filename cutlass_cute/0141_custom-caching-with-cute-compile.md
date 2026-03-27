---
title: "Custom Caching with cute.compile"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html#custom-caching-with-cute-compile"
---

### [Custom Caching with cute.compile](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#custom-caching-with-cute-compile)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#custom-caching-with-cute-compile "Permalink to this headline")

`cute.compile` bypasses caching in CuTe DSL and always performs compilation, returning a fixed JIT Executor instance.
This allows implementing custom caching strategies as shown below:

```python
@cute.jit
def add(b):
   return a + b

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # result = 4

# Use the custom cache
custom_cache[1](2) # result = 3
custom_cache[2](2) # result = 4
```
