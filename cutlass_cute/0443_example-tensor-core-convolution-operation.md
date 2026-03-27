---
title: "Example Tensor Core Convolution Operation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#example-tensor-core-convolution-operation"
---

## [Example Tensor Core Convolution Operation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#example-tensor-core-convolution-operation)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#example-tensor-core-convolution-operation "Permalink to this headline")

Example command line for profiling forward propagation convolution kernels runing on Tensor Cores is as follows:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=tensorop*fprop  --verification-providers=device --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_64x4_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=64 --stages=4  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.945071  ms
          Memory: 1114.21 GiB/s

            Math: 125369 GFLOP/s
```
