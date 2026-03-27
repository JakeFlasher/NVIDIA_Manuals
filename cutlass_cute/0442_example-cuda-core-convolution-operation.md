---
title: "Example CUDA Core Convolution Operation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#example-cuda-core-convolution-operation"
---

## [Example CUDA Core Convolution Operation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#example-cuda-core-convolution-operation)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#example-cuda-core-convolution-operation "Permalink to this headline")

Example command line for profiling forward propagation convolution kernels on CUDA cores is as follows:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=simt_sfprop  --verification-providers=device --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_simt_sfprop_optimized_128x128_8x2_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f32:nhwc --Filter=f32:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 2055798784  bytes
           FLOPs: 118482796544  flops

         Runtime: 8.13237  ms
          Memory: 235.431 GiB/s

            Math: 14569.3 GFLOP/s
```
