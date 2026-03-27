---
title: "Namespaces"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#namespaces"
---

#### [Namespaces](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#namespaces)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#namespaces "Permalink to this headline")

Namespaces are all lower case.
The top-level namespace is `cutlass::`.
The second nested namespace refers to
the general category of operation
performed by its members: e.g., `gemm::`.
The third nested namespace refers to
the operations’ position in the conceptual hierarchy:
e.g., `device::`, `kernel::`, or `collective::`.

The bodies of namespace definitions should not be indented.
Comments on the closing brace to indicate
the namespace being closed are welcome.

```c++
namespace cutlass {
namespace gemm {
namespace kernel {

struct AnotherGemmKernel {
  // ... contents ...
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
```
