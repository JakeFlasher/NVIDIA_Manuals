---
title: "Auto Kernel Dispatch Policies"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#auto-kernel-dispatch-policies"
---

### [Auto Kernel Dispatch Policies](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#auto-kernel-dispatch-policies)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#auto-kernel-dispatch-policies "Permalink to this headline")

In addition to direct dispatch policies listed above, the user can also use auto policies for both non-block scaled narrow-precision
GEMMs (both sparse and dense), and block scaled narrow-precision GEMMs (only dense).

CUTLASS will do its best to find the most efficient kernel for given parameters, however, the preferred method for building
these kernels is to use direct kernel dispatch policies shown in the above tables.

- `cutlass::gemm::collective::KernelScheduleAuto`: For a given Mma Tile Size, data type and layout combinations choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) and 1/2 SM `tcgen05.mma(.sp)`.
- `KernelTmaWarpSpecialized1SmBlockScaledSm100`: Use 1 SM `tcgen05.mma` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.
- `KernelTmaWarpSpecialized2SmBlockScaledSm100`: Use 2 SM `tcgen05.mma` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.
- `KernelSparseTmaWarpSpecialized1SmBlockScaledSm100`: Use 1 SM `tcgen05.mma.sp` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.
- `KernelSparseTmaWarpSpecialized2SmBlockScaledSm100`: Use 2 SM `tcgen05.mma.sp` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.

Similarly for epilogues, we can use `cutlass::epilogue::collective::EpilogueScheduleAuto`.
