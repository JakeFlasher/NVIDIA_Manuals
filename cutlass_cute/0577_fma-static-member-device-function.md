---
title: "fma static member device function"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#fma-static-member-device-function"
---

##### [fma static member device function](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#fma-static-member-device-function)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#fma-static-member-device-function "Permalink to this headline")

An operation struct defines a public `static void fma` function.
It is marked with the `CUTE_HOST_DEVICE` macro,
which adds the `__host__ __device__` annotations.
Different Operations define `fma` to take different numbers of arguments,
depending on the PTX MMA instruction.
The implementation protects use of the PTX instruction with a macro,
and raises an `assert` if `fma` is called when the macro is not defined.
This ensures that tests and examples that use this Operation in an Atom
can still compile, even if the PTX instruction is not available.
