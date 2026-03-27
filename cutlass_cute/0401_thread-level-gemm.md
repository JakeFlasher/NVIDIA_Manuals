---
title: "Thread-level GEMM"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#thread-level-gemm"
---

### [Thread-level GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#thread-level-gemm)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#thread-level-gemm "Permalink to this headline")

At the lowest level of blocking, each thread is responsible for processing a certain number of
elements. Threads cannot access each other’s registers, so we choose an organization that enables
reuse of values held in registers for multiple math instructions. This results in a 2D tiled
structure within a thread, in which each thread issues a sequence of independent math instructions
to the CUDA cores and computes an accumulated outer product.

SGEMM, IGEMM, HGEMM, and DGEMM are computed by SIMT math instructions issued by thread-level matrix multiply
procedures.
