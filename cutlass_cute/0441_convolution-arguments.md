---
title: "Convolution Arguments"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#convolution-arguments"
---

## [Convolution Arguments](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#convolution-arguments)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#convolution-arguments "Permalink to this headline")

```bash
$ ./tools/profiler/cutlass_profiler --help --operation=Conv2d

Conv2d

  [enum]      --conv_kind                                       Convolutional operator (fprop, dgrad, wgrad)
  [int]       --n,--input_n                                     Input N dimension of the Conv2d problem space
  [int]       --h,--input_h                                     Input H dimension of the Conv2d problem space
  [int]       --w,--input_w                                     Input W dimension of the Conv2d problem space
  [int]       --c,--input_c                                     Input C dimension of the Conv2d problem space
  [int]       --k,--filter_k                                    Filter K dimension of the Conv2d problem space
  [int]       --r,--filter_r                                    Filter R dimension of the Conv2d problem space
  [int]       --s,--filter_s                                    Filter S dimension of the Conv2d problem space
  [int]       --p,--output_p                                    Output P dimension of the Conv2d problem space
  [int]       --q,--output_q                                    Output Q dimension of the Conv2d problem space
  [int]       --g,--groups                                      Number of convolution groups
  [int]       --pad_h                                           Padding in H direction
  [int]       --pad_w                                           Padding in W direction
  [int]       --stride_h                                        Stride in H direction
  [int]       --stride_w                                        Stride in W direction
  [int]       --dilation_h                                      Dilation in H direction
  [int]       --dilation_w                                      Dilation in W direction
  [tensor]    --Activation                                      Tensor storing the Activation operand
  [tensor]    --Filter                                          Tensor storing the Filter operand
  [tensor]    --Output                                          Tensor storing the Output operand
  [enum]      --conv_mode                                       Convolution filter mode (conv, cross)
  [enum]      --iterator_algorithm,--iterator_algo              Convolution iterator algorithm (analytic, optimized)
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [enum]      --split_k_mode,--split-k-mode                     SplitK mode for serial or parallel reduction (serial, parallel)
  [int]       --split_k_slices,--split-k-slices                 Number of partitions of K dimension
  [enum]      --eq_gemm_provider,--eq-gemm-provider             Enable profiling equivalent gemm by the following providers (cutlass)
  [enum]      --op_class,--opcode-class                         Class of math instruction (simt, tensorop, wmmatensorop, wmma)
  [enum]      --accum,--accumulator-type                        Math instruction accumulator data type
  [int]       --cta_m,--threadblock-shape::m                    Threadblock shape in the M dimension
  [int]       --cta_n,--threadblock-shape::n                    Threadblock shape in the N dimension
  [int]       --cta_k,--threadblock-shape::k                    Threadblock shape in the K dimension
  [int]       --cluster_m,--cluster-shape::m                    Cluster shape in the M dimension
  [int]       --cluster_n,--cluster-shape::n                    Cluster shape in the N dimension
  [int]       --cluster_k,--cluster-shape::k                    Cluster shape in the K dimension
  [int]       --cluster_m_fallback,--cluster-shape-fallback::m  Fallback cluster shape in the M dimension
  [int]       --cluster_n_fallback,--cluster-shape-fallback::n  Fallback cluster shape in the N dimension
  [int]       --cluster_k_fallback,--cluster-shape-fallback::k  Fallback cluster shape in the K dimension
  [int]       --stages,--threadblock-stages                     Number of stages of threadblock-scoped matrix multiply
  [int]       --warps_m,--warp-count::m                         Number of warps within threadblock along the M dimension
  [int]       --warps_n,--warp-count::n                         Number of warps within threadblock along the N dimension
  [int]       --warps_k,--warp-count::k                         Number of warps within threadblock along the K dimension
  [int]       --inst_m,--instruction-shape::m                   Math instruction shape in the M dimension
  [int]       --inst_n,--instruction-shape::n                   Math instruction shape in the N dimension
  [int]       --inst_k,--instruction-shape::k                   Math instruction shape in the K dimension
  [int]       --min_cc,--minimum-compute-capability             Minimum device compute capability
  [int]       --max_cc,--maximum-compute-capability             Maximum device compute capability

Examples:

Profile a particular convolution (specify all the convolution parameters):
 $ cutlass_profiler --operation=Conv2d --Activation=f16:nhwc --Filter=f16:nhwc --Output=f16 --accumulator-type=f32 --n=32 --h=14 --w=14 --c=8 --k=64 --r=3 --s=3 --pad_h=1 --pad_w=1 --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1
```
