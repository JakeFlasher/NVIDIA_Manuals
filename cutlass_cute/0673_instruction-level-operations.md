---
title: "Instruction-level operations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#instruction-level-operations"
---

### [Instruction-level operations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#instruction-level-operations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#instruction-level-operations "Permalink to this headline")

CUTLASS defines a template-based interface to Tensor Core operations to avoid resorting
to inline PTX.

- [mma_sm70.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/mma_sm70.h) - Volta TensorCore operations
- [mma_sm75.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/mma_sm75.h) - Turing TensorCore operations
