---
title: "Numeric Conversion"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#numeric-conversion"
---

### [Numeric Conversion](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#numeric-conversion)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#numeric-conversion "Permalink to this headline")

CUTLASS defines procedures for performing numeric conversion between data types in `cutlass/numeric_conversion.h`.
Where possible, these target hardware acceleration on the target architecture and support multiple rounding modes.

```c++
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

NumericConverter<half_t, float>     convert_f32_to_f16;
NumericConverter<tfloat32_t, float> convert_f32_to_tf32;

half_t     x = convert_f32_to_f16(3.14159f);
tfloat32_t y = convert_f32_to_tf32(3.14159f);
```

Recent GPU architectures such as NVIDIA Turing and Ampere combine numeric conversion with efficient packing
into bit vectors. Consequently, CUTLASS defines conversion on both scalars and `Array<>` objects to implement
the optimal code sequence on all architectures.

```c++
//
// Example: convert and pack 32b signed integers to a vector of packed signed 8-bit integers.
//
int const kN = 16;
Array<int8_t, kN> destination;
Array<int,    kN> source;

NumericConverter<descltype(destination), decltype(source)> convert;

destination = convert(source);
```
