---
title: "5.4.2.3. Built-in Types"
section: "5.4.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-types"
---

### [5.4.2.3. Built-in Types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#built-in-types)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-types "Permalink to this headline")

CUDA provides vector types derived from basic integer and floating-point types that are supported for both the host and the device. The following table shows the available vector types.

| C++ Fundamental Type | Vector X1 | Vector X2 | Vector X3 | Vector X4 |
| --- | --- | --- | --- | --- |
| `signed char` | `char1` | `char2` | `char3` | `char4` |
| `unsigned char` | `uchar1` | `uchar2` | `uchar3` | `uchar4` |
| `signed short` | `short1` | `short2` | `short3` | `short4` |
| `unsigned short` | `ushort1` | `ushort2` | `ushort3` | `ushort4` |
| `signed int` | `int1` | `int2` | `int3` | `int4` |
| `unsigned` | `uint1` | `uint2` | `uint3` | `uint4` |
| `signed long` | `long1` | `long2` | `long3` | `long4_16a/long4_32a` |
| `unsigned long` | `ulong1` | `ulong2` | `ulong3` | `ulong4_16a/ulong4_32a` |
| `signed long long` | `longlong1` | `longlong2` | `longlong3` | `longlong4_16a/longlong4_32a` |
| `unsigned long long` | `ulonglong1` | `ulonglong2` | `ulonglong3` | `ulonglong4_16a/ulonglong4_32a` |
| `float` | `float1` | `float2` | `float3` | `float4` |
| `double` | `double1` | `double2` | `double3` | `double4_16a/double4_32a` |

Note that `long4`, `ulong4`, `longlong4`, `ulonglong4`, and `double4` have been deprecated in CUDA 13, and may be removed in a future release.

---

The following table details the byte size and alignment requirements of the vector types:

| Type | Size | Alignment |
| --- | --- | --- |
| `char1`, `uchar1` | 1 | 1 |
| `char2`, `uchar2` | 2 | 2 |
| `char3`, `uchar3` | 3 | 1 |
| `char4`, `uchar4` | 4 | 4 |
| `short1`, `ushort1` | 2 | 2 |
| `short2`, `ushort2` | 4 | 4 |
| `short3`, `ushort3` | 6 | 2 |
| `short4`, `ushort4` | 8 | 8 |
| `int1`, `uint1` | 4 | 4 |
| `int2`, `uint2` | 8 | 8 |
| `int3`, `uint3` | 12 | 4 |
| `int4`, `uint4` | 16 | 16 |
| `long1`, `ulong1` | 4/8 ***** | 4/8 ***** |
| `long2`, `ulong2` | 8/16 ***** | 8/16 ***** |
| `long3`, `ulong3` | 12/24 ***** | 4/8 ***** |
| `long4`, `ulong4` (deprecated) | 16/32 ***** | 16 ***** |
| `long4_16a`, `ulong4_16a` | 16/32 ***** | 16 |
| `long4_32a`, `ulong4_32a` | 16/32 ***** | 32 |
| `longlong1`, `ulonglong1` | 8 | 8 |
| `longlong2`, `ulonglong2` | 16 | 16 |
| `longlong3`, `ulonglong3` | 24 | 8 |
| `longlong4`, `ulonglong4` (deprecated) | 32 | 16 |
| `longlong4_16a`, `ulonglong4_16a` | 32 | 16 |
| `longlong4_32a`, `ulonglong4_32a` | 32 | 32 |
| `float1` | 4 | 4 |
| `float2` | 8 | 8 |
| `float3` | 12 | 4 |
| `float4` | 16 | 16 |
| `double1` | 8 | 8 |
| `double2` | 16 | 16 |
| `double3` | 24 | 8 |
| `double4` (deprecated) | 32 | 16 |
| `double4_16a` | 32 | 16 |
| `double4_32a` | 32 | 32 |

***** `long` is 4 bytes on C++ LLP64 data model (Windows 64-bit), while it is 8 bytes on C++ LP64 data model (Linux 64-bit).

---

Vector types are structures. Their first, second, third, and fourth components are accessible through the `x`, `y`, `z`, and `w` fields, respectively.

```cuda
int sum(int4 value) {
    return value.x + value.y + value.z + value.w;
}
```

They all have a factory function of the form `make_<type_name>()`; for example:

```cuda
int4 add_one(int x, int y, int z, int w) {
    return make_int4(x + 1, y + 1, z + 1, w + 1);
}
```

If host code is not compiled with `nvcc`, the vector types and related functions can be imported by including the `cuda_runtime.h` header provided in the CUDA toolkit.
