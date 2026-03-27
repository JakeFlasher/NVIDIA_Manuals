---
title: "MMA tile shapes supported"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#mma-tile-shapes-supported"
---

## [MMA tile shapes supported](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#mma-tile-shapes-supported)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#mma-tile-shapes-supported "Permalink to this headline")

The alignment restrictions also limit the options for Mma Tile Shapes. Tables below list the supported/valid `MmaTileShape`,
Layout, and Dispatch Policy combinations for each row of [Table 1](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#legacy_gemm_table), [Table 2](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#non_bs_gemm_table), and [Table 3](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#bs_gemm_table).

**Table 4: Valid Tile Shapes and Dispatch Policies for legacy types (All rows of Table 1)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x(4*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x64x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x192x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x64x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x192x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x(2/4*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 5: Valid Tile Shapes and Dispatch Policies for {float4_t, float6_t} x {float4_t, float6_t} (Rows 1,2,3,6,10,11,12,and 15 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | N | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | N | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | N | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | N | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x128x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x128x128 | N | N | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 6: Valid Tile Shapes and Dispatch Policies for float8_t x {float4_t, float6_t} (Rows 5,8,14,and 17 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x128x128 | Y | Y | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 7: Valid Tile Shapes and Dispatch Policies for {float4_t, float6_t} x float8_t (Rows 4,7,13,and 16 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x128x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x128x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 8: Valid Tile Shapes and Dispatch Policies for float8_t x float8_t (Row 9,18 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x64x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x64x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 9: Valid Tile Shapes for nv_float4_t x nv_float4_t (Row 1 and 12 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| Dense | 1SM | 128x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| Dense | 1SM | 128x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| Dense | 2SM | 256x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| Dense | 2SM | 256x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| Dense | 2SM | 256x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |

**Table 10: Valid Tile Shapes and Dispatch Policies for mx_float4_t x mx_float4_t (Row 2 and 13 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmMxf4Sm100` |
| Dense | 1SM | 128x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmMxf4Sm100` |
| Dense | 1SM | 128x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmMxf4Sm100` |
| Dense | 2SM | 256x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| Dense | 2SM | 256x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| Dense | 2SM | 256x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |

**Table 11: Valid Tile Shapes and Dispatch Policies for mx_float4_t x mx_float4_t (Row 3 and 14 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x192x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |

**Table 12: Valid Tile Shapes and Dispatch Policies for {mx_float4_t, mx_float6_t, mx_float8_t} x {mx_float4_t, mx_float6_t} (Rows 4, 5, 7, 8, 10, 15, 16, 18, 19, and 21 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x192x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |

**Table 13: Valid Tile Shapes and Dispatch Policies for {mx_float4_t, mx_float6_t, mx_float8_t} x mx_float8_t (Rows 6, 9, 11, 17, 20, and 22 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
