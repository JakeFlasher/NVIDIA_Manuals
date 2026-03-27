---
title: "Using CUTLASS within other applications"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#using-cutlass-within-other-applications"
---

## [Using CUTLASS within other applications](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#using-cutlass-within-other-applications)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#using-cutlass-within-other-applications "Permalink to this headline")

Applications should list [`/include`](https://github.com/NVIDIA/cutlass/tree/main/include) within their include paths. They must be
compiled as C++17 or greater.

**Example:** print the contents of a variable storing half-precision data.

```c++
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main() {

  cutlass::half_t x = 2.25_hf;

  std::cout << x << std::endl;

  return 0;
}
```
