---
title: "@kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#kernel"
---

### [@kernel](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#kernel "Permalink to this headline")

Defines GPU kernel functions, compiled as specialized GPU symbols through dynamic compilation.

**Decorator Parameters**:

- `preprocessor`:
  - `True` (default) — Automatically expands Python loops/ifs into GPU-compatible IR operations.
  - `False` — Expects manual or simplified kernel implementations.

**Kernel Launch Parameters**:

- `grid`
Specifies the grid size as a list of integers.
- `block`
Specifies the block size as a list of integers.
- `cluster`
Specifies the cluster size as a list of integers.
- `smem`
Specifies the size of shared memory in bytes (integer).
