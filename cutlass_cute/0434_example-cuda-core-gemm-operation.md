---
title: "Example CUDA Core GEMM Operation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#example-cuda-core-gemm-operation"
---

## [Example CUDA Core GEMM Operation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#example-cuda-core-gemm-operation)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#example-cuda-core-gemm-operation "Permalink to this headline")

Example command line for profiling SGEMM kernels is as follows:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 115992428544  flops

         Runtime: 6.73655  ms
          Memory: 24.934 GiB/s

            Math: 17218.4 GFLOP/s
```

Note, the arguments which appear in the output may be used as command line parameters for subsequent invocations.
