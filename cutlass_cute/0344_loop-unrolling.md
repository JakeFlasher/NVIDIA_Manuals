---
title: "Loop Unrolling"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#loop-unrolling"
---

### [Loop Unrolling](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#loop-unrolling)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#loop-unrolling "Permalink to this headline")

CUTLASS requires tiles of data to be stored in registers for high-bandwidth access. Simultaneously, high-throughput math instructions
must be issued concurrently with memory instructions to hide latency with relatively few concurrent threads. These objectives are
achieved by unrolling loops whose iteration counts are known at compile time.

Consequently, most loops within the CUTLASS GEMM implementation are specified by constant values and template arguments. The CUDA compiler
is able to unroll the loop bodies, map array elements to registers, and construct an efficient instruction schedule.

All loops expected to be unrolled should be annotated with `CUTLASS_PRAGMA_UNROLL` to explicitly direct the compiler
to unroll them.

```c++
int const kN = 8;
Array<float, kN> x;                       // Array we would like to store in registers

CUTLASS_PRAGMA_UNROLL                     // Directs the CUDA compiler to unroll this loop.
for (int idx = 0; idx < kN; ++idx) {      // Loop has constant number of iterations.

  x[i] = float(idx);                      // Indirect access by induction variable results in
                                          // direct register access.
}
```
