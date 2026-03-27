---
title: "Reference Implementations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#reference-implementations"
---

## [Reference Implementations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#reference-implementations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#reference-implementations "Permalink to this headline")

CUTLASS defines reference implementations usable with all data types and layouts. These are
used throughout the unit tests.

**Example:** Reference GEMM implementation with mixed precision internal computation.

```c++
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>

int main() {

  int M = 64;
  int N = 32;
  int K = 16;

  float alpha = 1.5f;
  float beta = -1.25f;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::reference::host::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementA and LayoutA
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementB and LayoutB
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementC and LayoutC
    float,                                           // scalar type (alpha and beta)
    float> gemm_op;                                  // internal accumulation type

  gemm_op(
    {M, N, K},             // problem size
    alpha,                 // alpha scalar
    A.host_view(),         // TensorView to host memory
    B.host_view(),         // TensorView to host memory
    beta,                  // beta scalar
    C.host_view(),         // TensorView to host memory
    D.host_view());        // TensorView to device memory

  return 0;
}
```
