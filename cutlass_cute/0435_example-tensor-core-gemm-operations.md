---
title: "Example Tensor Core GEMM Operations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#example-tensor-core-gemm-operations"
---

## [Example Tensor Core GEMM Operations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#example-tensor-core-gemm-operations)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#example-tensor-core-gemm-operations "Permalink to this headline")

To execute kernels targeting Tensor Core operations, supply the flag `--op_class=tensorop` in the command line.

```bash
$ ./tools/profiler/cutlass_profiler --op_class=tensorop --m=3456 --n=4096 --k=8192

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s16816gemm_f16_256x128_32x3_nn_align8

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=8192 --A=f16:column --B=f16:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128 --cta_k=32 --stages=3 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 231956545536  flops

         Runtime: 0.98647  ms
          Memory: 170.272 GiB/s

            Math: 235138 GFLOP/s
```
