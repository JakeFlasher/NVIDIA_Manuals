---
title: "Minimizing Host Overhead"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html#minimizing-host-overhead"
---

## [Minimizing Host Overhead](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#minimizing-host-overhead)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#minimizing-host-overhead "Permalink to this headline")

Eager kernel invocation overhead on the CPU host can sometimes become a bottleneck
for latency-sensitive applications. TVM FFI can help greatly reduce this overhead.
To maximize performance benefits, we recommend setting up your workflow as follows
(detailed instructions are provided in subsequent sections):

- **Compile the kernel with TVM FFI enabled.**
- **Declare shape constraints using fake tensors** and reuse the compiled function
throughout your execution.
- **Pass PyTorch tensors directly** to the compiled function to avoid explicit DLPack conversion.
- **Use the environment stream flag** to implicitly pass the current PyTorch stream.
- **Rely on compiled argument validation** instead of Python-side attribute validation,
as TVM FFI functions perform fast compiled checks.

Following these steps can significantly reduce the host-side overhead of eager kernel execution.
The sections below provide detailed examples and explanations for each step.
You may find it helpful to refer back to this summary after you review the implementation details.
