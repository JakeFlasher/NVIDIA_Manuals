---
title: "Blackwell Narrow Precision Data Types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-narrow-precision-data-types"
---

### [Blackwell Narrow Precision Data Types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#blackwell-narrow-precision-data-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#blackwell-narrow-precision-data-types "Permalink to this headline")

Narrow-precision `tcgen05.mma` instructions can operate on several 4, 6, and 8-bit data types. Blackwell MMAs can operate
on five different 8-bit floating point values, of which only two (`float_ue8m0_t` and `float_ue4m3_t`) can be used as scale factor data types.
There are two 6-bit floating point types and one 4-bit floating point data type.
See [PTX documentation for narrow precision data types](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-data-formats) for details.

**Blackwell Narrow Precision Data Types**

| Data Type | Exponent Bits | Mantissa Bits | Signed | Bit Size |
| --- | --- | --- | --- | --- |
| float_e4m3_t | 4 | 3 | Yes | 8 |
| float_e5m2_t | 5 | 2 | Yes | 8 |
| float_e2m3_t | 2 | 3 | Yes | 6 |
| float_e3m2_t | 3 | 2 | Yes | 6 |
| float_e2m1_t | 2 | 1 | Yes | 4 |
| float_ue8m0_t[^[1]] | 8 | 0 | No | 8 |
| float_ue4m3_t[^[1]] | 4 | 3 | No | 8 |

Block scaled MMAs use `mx` and `nv` types which are a pair of float8_t, float6_t, float4_t with 2 of the scale factor data types with a predetermined scale factor vector size. `mx` types follow OCP specification (see [OCP Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)). The following types provided by CUTLASS can be used as inputs to collective builders to generate the block scaled kernels:

**Blackwell Block Scaled Narrow Precision Data Types**

| Mx/Nv Data Type | Scale Factor Type | SF Vector Size (Dense) | SF Vector Size (Sparse) | OCP Compliant |
| --- | --- | --- | --- | --- |
| mx_float8_t<Any F8type> | float_ue8m0_t | 32 | 64 | Yes |
| mx_float6_t<Any F6Type> | float_ue8m0_t | 32 | 64 | Yes |
| mx_float4_t | float_ue8m0_t | 32 | 64 | Yes |
| nv_float4_t | float_ue4m3_t | 16 | 32 | No |
