---
title: "Calling Conventions"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#calling-conventions"
---

## [Calling Conventions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#calling-conventions)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#calling-conventions "Permalink to this headline")

| **Caller** | **Callee** | **Allowed** | **Compilation/Runtime** |
| --- | --- | --- | --- |
| Python function | `@jit` | ✅ | DSL runtime |
| Python function | `@kernel` | ❌ | N/A (error raised) |
| `@jit` | `@jit` | ✅ | Compile-time call, inlined |
| `@jit` | Python function | ✅ | Compile-time call, inlined |
| `@jit` | `@kernel` | ✅ | Dynamic call via GPU driver or runtime |
| `@kernel` | `@jit` | ✅ | Compile-time call, inlined |
| `@kernel` | Python function | ✅ | Compile-time call, inlined |
| `@kernel` | `@kernel` | ❌ | N/A (error raised) |
