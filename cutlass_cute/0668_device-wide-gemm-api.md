---
title: "Device-wide GEMM API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#device-wide-gemm-api"
---

### [Device-wide GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#device-wide-gemm-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#device-wide-gemm-api "Permalink to this headline")

The device-level GEMM API is intended to streamline instantiation and execution of the standard
GEMM computation across the GPU. This operator is intended to be used in host-side .cu code and
has semantics similar to cuBLAS.

The device-wide GEMM API is embodied by the following operators:

- [cutlass::gemm::device::Gemm](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm.h) - basic GEMM operation
- [cutlass::gemm::device::GemmArray](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_array.h) - batched GEMM operation in which input matrices are read from arrays of pointers
- [cutlass::gemm::device::GemmBatched](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_batched.h) - batched GEMM operation in which input matrices are separated by a constant stride
- [cutlass::gemm::device::GemmSplitKParallel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_splitk_parallel.h) - GEMM operation that partitions the GEMM K dimension then launches a separate reduction kernel

**Example:** launch a mixed-precision GEMM targeting Volta Tensor Cores.

```c++
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm70                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Launch GEMM on the device
  //

  status = gemm_op({
    {m, n, k},
    {ptrA, lda},
    {ptrB, ldb},
    {ptrC, ldc},
    {ptrD, ldd},
    {alpha, beta}
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }
```
