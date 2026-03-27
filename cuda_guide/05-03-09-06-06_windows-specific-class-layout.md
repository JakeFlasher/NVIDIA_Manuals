---
title: "5.3.9.6.6. Windows-Specific Class Layout"
section: "5.3.9.6.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#windows-specific-class-layout"
---

#### [5.3.9.6.6. Windows-Specific Class Layout](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#windows-specific-class-layout)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#windows-specific-class-layout "Permalink to this headline")

The CUDA compiler follows the IA64 ABI for class layout, while Microsoft Visual Studio does not. This prevents bitwise copy of special objects between host and device code as described below.

Let `T` denote a pointer to member type, or a class type that satisfies any of the following conditions:

- `T` is a [polymorphic class](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#polymorphic-classes)
- `T` has multiple inheritance with more than one direct or indirect [empty base class](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#class-type-variables).
- All direct and indirect base classes `B` are [empty](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#class-type-variables) and the type of the first field `F` of `T` uses `B` in its definition, such that `B` is laid out at offset 0 in the definition of `F`.

Classes of type `T`, with a base class of type `T`, or with data members of type `T`, may have a different class layout and size between host and device when compiled with Microsoft Visual Studio.

Copying such objects from device to host or from host to device, including `__global__` function arguments is undefined behavior.
