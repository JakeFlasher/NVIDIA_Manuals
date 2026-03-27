---
title: "Features"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html#features"
---

### [Features](https://docs.nvidia.com/cuda/cutile-python/generated#features)[](https://docs.nvidia.com/cuda/cutile-python/generated/#features "Permalink to this headline")

- Add `ct.static_iter` keyword that enables compile-time `for` loops.
- Add `ct.static_assert` keyword that can be used to assert that a condition is true at compile time.
- Add `ct.static_eval` keyword that enables compile-time evaluation using the host Python interpreter.
- Add `ct.scan()` for custom scan.
- Add `ct.isnan()`.
- Add `print()` and `ct.print()` that supports python-style print and f-strings.
- Add optional `mask` parameter to `ct.gather()` and `ct.scatter()` for custom boolean masking.
- Operator `+` can now be used to concatenate tuples.
- Support unpacking nested tuples (e.g., `a, (b, c) = t`) and using square brackets
for unpacking (e.g., `[a, b] = 1, 2`).
- Add bytecode-to-cubin disk cache to avoid recompilation of unchanged kernels.
Controlled by `CUDA_TILE_CACHE_DIR` and `CUDA_TILE_CACHE_SIZE`.
