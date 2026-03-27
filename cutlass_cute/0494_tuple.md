---
title: "Tuple"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#tuple"
---

### [Tuple](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#tuple)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#tuple "Permalink to this headline")

A tuple is a finite ordered list of zero or more elements.
The [`cute::tuple` class](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container/tuple.hpp) behaves like `std::tuple`, but works on device and host. It imposes restrictions on its template arguments and strips down the implementation for performance and simplicity.
