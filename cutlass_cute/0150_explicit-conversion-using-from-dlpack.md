---
title: "Explicit conversion using from_dlpack"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#explicit-conversion-using-from-dlpack"
---

## [Explicit conversion using from_dlpack](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#explicit-conversion-using-from-dlpack)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#explicit-conversion-using-from-dlpack "Permalink to this headline")

CuTe DSL’s runtime provides an interface for converting DLPack-compatible tensors to CuTe tensors,

```python
b = cute.runtime.from_dlpack(a)
```

where `a` is a tensor supporting the DLPack protocol with the `__dlpack__`
and `__dlpack_device__` methods. The resulting CuTe tensor `b` has a fully static layout. This
conversion is performed without copying any tensor data, enabling seamless integration with major
frameworks. Users can create tensors using NumPy, PyTorch, etc. and directly feed them into JIT
functions writtnen using CuTe DSL.

The resulting CuTe tensor shares the same underlying memory buffer as the original tensor. This
zero-copy approach maximizes performance by eliminating unnecessary data duplication. However, it is
important to note that the CuTe tensor’s validity is tied to the lifetime of the original tensor. If
the source tensor is destroyed or goes out of scope, the corresponding CuTe tensor becomes invalid
since it references the original memory location.

The full signature of from_dlpack is as follows:

```python
def from_dlpack(tensor, assumed_align=None, use_32bit_stride=False):
```

The `assumed_align` integer parameter specifies the alignment of the tensor in unit of bytes.
The tensor’s base address must be divisible by `assumed_align`. When not provided explicitly,
the alignment is set to the natural alignment of the tensor’s element type. Note that the alignment
information is part of the pointer type in the generated IR. Therefore, programs with different
alignments have a different IR and identical IRs are required for hitting the kernel caching
mechanism of CuTe DSL.

The `use_32bit_stride` parameter determines whether to use 32-bit stride for the tensor’s dynamic stride values.
By default, it is set to False (64bit) to ensure that address calculations do not risk overflow. For smaller
problem sizes (where `cosize(layout_of_tensor) <= Int32_MAX`), users may set it to True (32bit) to improve performance
by reducing register usage and the number of address calculation instructions. When `use_32bit_stride` is set
to True, a runtime check is performed to ensure that the layout does not overflow. Please note that this parameter
only has an effect when the tensor’s layout is marked as dynamic.
