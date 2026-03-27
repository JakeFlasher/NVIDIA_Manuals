---
title: "GEMM Arguments"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#gemm-arguments"
---

## [GEMM Arguments](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#gemm-arguments)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#gemm-arguments "Permalink to this headline")

The complete set of arguments available to each operation may be viewed by specifying the operation name
in addition to `--help`. The argument flags and their aliases usable for GEMM appear as follows.

```bash
$ ./tools/profiler/cutlass_profiler --operation=gemm --help

GEMM

  [enum]      --gemm_kind                                       Variant of GEMM (e.g. universal, gemm, planar_complex, planar_complex_array)
  [int]       --m,--problem-size::m                             M dimension of the GEMM problem space
  [int]       --n,--problem-size::n                             N dimension of the GEMM problem space
  [int]       --k,--problem-size::k                             K dimension of the GEMM problem space
  [tensor]    --A                                               Tensor storing the A operand
  [tensor]    --B                                               Tensor storing the B operand
  [tensor]    --C                                               Tensor storing the C operand
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [enum]      --split_k_mode,--split-k-mode                     Variant of split K mode(serial, parallel)
  [int]       --split_k_slices,--split-k-slices                 Number of partitions of K dimension
  [int]       --batch_count,--batch-count                       Number of GEMMs computed in one batch
  [enum]      --op_class,--opcode-class                         Class of math instruction (simt, tensorop, wmmatensorop, wmma).
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
  [enum]      --raster_order={heuristic|H|along_m|M|along_n|N}  If supported by kernel, sets the tile raster direction
  [int]       --swizzle_size={1,2,4,8}                          If supported by kernel, sets the 2D tile swizzle extent (In Hopper, other values will be rounded down to the nearest supported value)
  [int]       --use_pdl,--use-pdl                               Use PDL (true, false)
  [int]       --enable_sm90_mixed_dtype_shuffle_test            If true, the profiler will test SM90 mixed input kernels that can use shuffled input layouts for better performance
  [enum]      --runtime_input_datatype_a                        Runtime data type for A matrix, narrow-precision only (e4m3, e5m2, e3m2, e2m3, e2m1)
  [enum]      --runtime_input_datatype_b                        Runtime data type for B matrix, narrow-precision only (e4m3, e5m2, e3m2, e2m3, e2m1)

Examples:

Profile a particular problem size:
  $ cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=128

Schmoo over problem size and beta:
  $ cutlass_profiler --operation=Gemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5

Schmoo over accumulator types:
  $ cutlass_profiler --operation=Gemm --accumulator-type=f16,f32

Run when A is f16 with column-major and B is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):
  $ cutlass_profiler --operation=Gemm --A=f16:column --B=*:row

Using various input value distribution:
  $ cutlass_profiler --operation=Gemm --dist=uniform,min:0,max:3
  $ cutlass_profiler --operation=Gemm --dist=gaussian,mean:0,stddev:3
  $ cutlass_profiler --operation=Gemm --dist=sequential,start:0,delta:1

Using CUTLASS 3.x GEMM kernel with a tile scheduler that supports runtime tile remapping and raster mode order:
  $ cutlass_profiler --operation=Gemm --m=2048 --n=2048 --k=2048 --raster_order=M --swizzle_size=2

Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):
 $ cutlass_profiler --operation=Gemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect

Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:
 $ cutlass_profiler  --operation=Gemm \
   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --k=8,16,32,64,128,256,288,384,504,512,520 \
   --beta=0,1,2 --profiling-iterations=1 \
   --providers=cutlass --output=functional-test.csv

Profile when execution is performed on device 0 and the C tensor is located on a device 1 and D on device 2:
  $ cutlass_profiler --device=0 --allocations=C:1,D:2 --operation=Gemm --m=1024 --n=1024 --k=128
```

The format of tensor argument is followed by `<type>:<layout>`. The type could be `f32` as 32-bit floating point, `s8` as 8-bit signed integer, etc. The available types can be referred to the `NumericTypeID_enumerants` in [util.cu](https://github.com/NVIDIA/cutlass/tree/main/tools/library/src/util.cu). The layout could be `row` or `column`. If `--enable_sm90_mixed_dtype_shuffle_test=true` is used, the actual layout of the narrow data type matrix is a shuffled layout, neither `row` nor `column`.

In addition to encoded data types, CUTLASS profiler allows non-encoded generic data types, namely `f8`, `f6`, and `f4`, with corresponding encoding specified through GEMM input argument: `--runtime_input_datatype_a` and `--runtime_input_datatype_b`. Currently, six encoding schemes are supported: `e4m3`, `e5m2`, `e3m2`, `e2m3`, and `e2m1`.

Cluster shapes can be statically set to `Shape<int,int,_1>;` and specified via runtime arguments: `cluster_m`, `cluster_n` and `cluster_k` in CUTLASS profiler.  In addition to preferred cluster shapes, a user can also specify fallback cluster shapes via runtime arguments: `cluster_m_fallback`, `cluster_n_fallback` and `cluster_k_fallback` in CUTLASS profiler. Those fallback cluster shapes are smaller shapes than the preferred ones for the hardware to assign when there is no chance to issue a larger preferred CGA cluster to the GPU. There are several rules for using a flexible CGA: 1) Preferred CGA size should be divisible by fallback CGA size. 2) Grid dim should be divisible by preferred CGA size. 3) Preferred CGA and fallback CGA must have the same depth (cluster_dim.z must be equal). One may refer to our CUTLASS Example [73_blackwell_gemm_flexible_cluster](https://github.com/NVIDIA/cutlass/tree/main/examples/73_blackwell_gemm_preferred_cluster/blackwell_gemm_preferred_cluster.cu) for more details of the this feature.
Please be noted that this feature (flexible cluster shapes within a single grid) is only applicable to `sm100a` kernels. The hardware will rasterize into a single cluster shape for those kernels that do not support this feature even with preferred or fallback cluster shapes assigned.

CUTLASS 3.x kernels for Hopper and Blackwell also support a new feature called programatic dependent launch (PDL). This can be enabled with `--use-pdl`, and can overlap the epilogue of the prior kernel with the prologue of the next kernel. This can effectively hide kernel prologues. Using PDL can improve performance for back to back GEMMs. See [dependent kernel launch](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html) for more information. CUDA graphs can also be used (`--use-cuda-graphs`) with PDL to ensure that smaller kernels are enqueued back-to-back on a stream.
