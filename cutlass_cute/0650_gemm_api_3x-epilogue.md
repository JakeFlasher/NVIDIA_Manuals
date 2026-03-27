---
title: "Epilogue"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#gemm_api_3x--epilogue"
---

### [Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#epilogue)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#epilogue "Permalink to this headline")

The collective epilogue implements element-wise operations
involving the output matrix.  Users can provide a custom
epilogue, or use one of the standard epilogues.
These live in the directory
[include/cutlass/epilogue/collective/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/),
and include classes like
`cutlass::epilogue::collective::DefaultEpilogue`
and
`cutlass::epilogue::collective::Epilogue`.
CUTLASS’s provided collective epilogues
do not live under `include/cutlass/gemm`
or in the `cutlass::gemm` namespace,
because they can be used for computations
other than GEMM.
