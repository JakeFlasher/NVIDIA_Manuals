---
title: "Tile size:"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#tile-size"
---

## [Tile size:](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tile-size)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tile-size "Permalink to this headline")

Below are tables that summarize the valid tile shapes and dispatch policies for SM120 GEMMs. If the output is `float_6_t`, the tile size in the leading dimension of output tensor must be 128.

**Table 16: Valid Tile Shapes and Dispatch Policies for {float8_t, float_6_t, float_4_t} x {float8_t, float_6_t, float_4_t} of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 64x64x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 64x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 128x64x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

**Table 17: Valid Tile Shapes for nv_float4_t x nv_float4_t of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 256x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedCooperative` |
| 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

**Table 18: Valid Tile Shapes and Dispatch Policies for mx_float4_t x mx_float4_t of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 256x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedCooperative` |
| 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

**Table 19: Valid Tile Shapes and Dispatch Policies for mx_float4_t x mx_float4_t of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedMxf8f6f4Sm120` or `KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120` |
| 256x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedMxf8f6f4Sm120` |
| 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecializedMxf8f6f4Sm120` or `KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120` |

Specialized policies must be used to generate mixed-input-datatype `mx_float4_t` kernels.

**Table 20: Valid Tile Shapes and Dispatch Policies for {mx_float4_t, mx_float6_t, mx_float8_t} x {mx_float4_t, mx_float6_t, mx_float8_t}**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
