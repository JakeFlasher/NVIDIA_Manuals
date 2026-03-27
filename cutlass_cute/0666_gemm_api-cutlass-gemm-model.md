---
title: "CUTLASS GEMM Model"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#gemm_api--cutlass-gemm-model"
---

## [CUTLASS GEMM Model](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-gemm-model)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-gemm-model "Permalink to this headline")

CUTLASS implements the basic GEMM triple loop nest with a tiled structure mirroring the execution model hierarchy.

The following pseudocode describes the model for a GEMM kernel targeting a warp-synchronous matrix multiply instruction like
mma.sync. The entire operation is referred to as “Gemm,” as it is assumed that an epilogue operation performs the general matrix
update similar to BLAS.

```c++
                                                                            // cutlass::gemm::device::Gemm
                                                                            //
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each CTA       } CTA-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each CTA    }
                                                                            //
                                                                            // cutlass::gemm::threadblock::Mma
                                                                            //
    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp      } warp-level concurrency
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp   }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {    //       fully unroll across CtaTileK - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // cutlass::gemm::warp::Mma
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //
                                                                            //
                  mma_instruction(d, a, b, c);                              // cutlass::arch::mma - warp-wide matrix multiply instruction

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```

The outer-most loops correspond to CTA-level hardware concurrency and are not explicitly written as loops in the code. These
are implied by CUDA grid launch semantics.

The comment `cutlass::gemm::threadblock::Mma` refers to the threadblock-scoped matrix multiply-accumulate concept. This is
the computation performed by one threadblock to compute a matrix product in registers. The “GEMM main loop” is listed.

The comment `cutlass::gemm::warp::Mma` refers to the computation performed by each warp. This is a nested loop executing a
sequence of accumulated outer products.

The inner-most operation corresponds directly to hardware support. In this example, the nested structure terminates with
warp-synchronous matrix multiply instructions targeting Tensor Cores.
Alternatively, GEMMs targeting single-thread instructions may have an additional series of nested loops corresponding to
thread-level concurrency.
