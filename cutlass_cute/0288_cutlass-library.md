---
title: "CUTLASS Library"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#cutlass-library"
---

## [CUTLASS Library](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-library)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-library "Permalink to this headline")

The [CUTLASS Library](https://github.com/NVIDIA/cutlass/tree/main/tools/library) defines an API for managing and executing collections of compiled
kernel instances and launching them from host code without template instantiations in client code.

The host-side launch API is designed to be analogous to BLAS implementations for convenience, though its
kernel selection procedure is intended only to be functionally sufficient. It may not launch the
optimal tile size for a given problem. It chooses the first available kernel whose data types,
layouts, and alignment constraints satisfy the given problem. Kernel instances and a data structure
describing them are completely available to client applications which may choose to implement their
own selection logic.

[cuBLAS](https://developer.nvidia.com/cublas) offers the best performance and functional coverage
for dense matrix computations on NVIDIA GPUs.

The CUTLASS Library is used by the CUTLASS Profiler to manage kernel instances, and it is also used
by several SDK examples.

- [10_planar_complex](https://github.com/NVIDIA/cutlass/tree/main/examples/10_planar_complex/planar_complex.cu)
- [11_planar_complex_array](https://github.com/NVIDIA/cutlass/tree/main/examples/11_planar_complex_array/planar_complex_array.cu)

The CUTLASS Library defines enumerated types describing numeric data types, matrix and tensor
layouts, math operation classes, complex transformations, and more.

Client applications should specify [`tools/library/include`](https://github.com/NVIDIA/cutlass/tree/main/tools/library/include) in their
include paths and link against libcutlas_lib.so.

The CUTLASS SDK example [10_planar_complex](https://github.com/NVIDIA/cutlass/tree/main/examples/10_planar_complex/CMakeLists.txt) specifies
its dependency on the CUTLASS Library with the following CMake command.

```console
target_link_libraries(
  10_planar_complex
  PRIVATE
  cutlass_lib
  cutlass_tools_util_includes
)
```

A sample kernel launch from host-side C++ is shown as follows.

```c++
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

int main() {

  //
  // Define the problem size
  //
  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> C({M, N});

  float const *ptrA = A.device_data();
  float const *ptrB = B.device_data();
  float const *ptrC = C.device_data();
  float       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = D.device_ref().stride(0);

  //
  // CUTLASS Library call to execute device GEMM
  //

  cutlass::library::Handle handle;

  //
  // Launch GEMM on CUDA device.
  //

  cutlass::Status status = handle.gemm(
    M,
    N,
    K,

    cutlass::library::NumericTypeID::kF32,          // data type of internal accumulation
    cutlass::library::NumericTypeID::kF32,          // data type of alpha/beta scalars

    &alpha,                                         // pointer to alpha scalar

    cutlass::library::NumericTypeID::kF32,          // data type of A matrix
    cutlass::library::LayoutTypeID::kColumnMajor,   // layout of A matrix
    ptrA,                                           // pointer to A matrix in device memory
    lda,                                            // leading dimension of A matrix

    cutlass::library::NumericTypeID::kF32,          // data type of B matrix
    cutlass::library::LayoutTypeID::kColumnMajor,   // layout of B matrix
    ptrB,                                           // pointer to B matrix in device memory
    ldb,                                            // leading dimension of B matrix

    &beta,                                          // pointer to beta scalar

    cutlass::library::NumericTypeID::kF32,          // data type of C and D matrix

    ptrC,                                           // pointer to C matrix in device memory
    ldc,                                            // leading dimension fo C matrix

    ptrD,                                           // pointer to D matrix in device memory
    ldd                                             // leading dimension of D matrix
  );

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  return 0;
}
```
