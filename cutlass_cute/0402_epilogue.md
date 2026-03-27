---
title: "Epilogue"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#epilogue"
---

## [Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#epilogue)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#epilogue "Permalink to this headline")

The above code focuses only on the matrix multiply computation **C = AB** whose result is
held in the registers of each thread within the threadblock. The mapping of logical elements
in the output tile to each thread is chosen to maximize performance of the matrix multiply
computation but does not result in efficient, coalesced loads and stores to global memory.

The epilogue is a separate phase in which threads exchange data through shared memory then
cooperatively access global memory using efficient striped access patterns. It is also
the phase in which linear scaling and other elementwise operations may be conveniently
computed using the matrix product results as inputs.

CUTLASS defines several typical epilogue operations such as linear scaling and clamping,
but other device-side function call operators may be used to perform custom operations.
