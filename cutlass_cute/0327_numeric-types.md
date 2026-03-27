---
title: "Numeric Types"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#numeric-types"
---

## [Numeric Types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#numeric-types)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#numeric-types "Permalink to this headline")

CUTLASS defines classes for the following numeric data types.

- `half_t`: IEEE half-precision floating point (exponent: 5b, mantissa: 10b; literal suffix `_hf`)
- `bfloat16_t`: BFloat16 data type (exponent: 8b, mantissa: 7b; literal suffix `_bf16`)
- `tfloat32_t`: Tensor Float 32 data type (exponent: 8b, mantissa: 10b; literal suffix `_tf32`)
- `int4_t`, `uint4_t`: 4b signed and unsigned integer (literal suffx `_s4`, `_u4`)
- `bin1_t`: 1b binary numeric type (literal suffix `_b1`)
- `float_e5m2_t`: 8bits signed float (exponent: 5 bits, mantissa: 2 bits)
- `float_e4m3_t`: 8bits signed float (exponent: 4 bits, mantissa: 3 bits)
- `float_ue4m3_t`: 8bits unsigned float (exponent: 4 bits, mantissa: 3 bits)
- `float_ue8m0_t`: 8bits unsigned float (exponent: 8 bits, mantissa: 0 bits)
- `float_e3m2_t`: 6bits signed float (exponent: 3 bits, mantissa: 2 bits)
- `float_e2m3_t`: 6bits signed float (exponent: 2 bits, mantissa: 3 bits)
- `float_e2m1_t`: 4bits signed float (exponent: 2 bits, mantissa: 1 bits)
- `type_erased_dynamic_float8_t`: Type agnostic 8 bits signed float allowing the user to provide a specific datatype as runtime argument.
- `type_erased_dynamic_float6_t`: Type agnostic 6 bits signed float allowing the user to provide a specific datatype as runtime argument.
- `type_erased_dynamic_float4_t`: Type agnostic 4 bits signed float allowing the user to provide a specific datatype as runtime argument.
- `mx_float8_t<float_e5m2_t>` or `mx_float8_t<float_e4m3_t>` : Block scaled data type with fp8 element type and float_ue8m0_t scale factor and vector size of 32.
- `mx_float6_t<float_e3m2_t>` or `mx_float6_t<float_e2m3_t>` : Block scaled data type with fp6 element type and float_ue8m0_t scale factor and vector size of 32.
- `mx_float4_t<float_e2m1_t>` : Block scaled data type with signed e2m1 element type and float_ue8m0_t scale factor and vector size of 32.
- `nv_float4_t<float_e2m1_t>` : Block scaled data type with signed e2m1 element type and float_ue4m3_t scale factor and vector size of 16.
- `complex<T>`: defines complex-valued data type based on the supplied real-valued numeric type

Numeric types in CUTLASS may be used in both host and device code and are intended to function
like any other plain-old-data type.

If CUTLASS is compiled with `CUTLASS_F16C_ENABLED`, then hardware conversion is used for
half-precision types in host code. Regardless, `cutlass::half_t` uses the most efficient
NVIDIA GPU hardware instructions available in device code.

Example:

```c++
#include <iostream>
#include <cutlass/numeric_types.h>

__global__ void kernel(cutlass::half_t x) {
  printf("Device: %f\n", float(x * 2.0_hf));
}

int main() {

  cutlass::half_t x = 0.5_hf;

  std::cin >> x;

  std::cout << "Host: " << 2.0_hf * x << std::endl;

  kernel<<< dim3(1,1), dim3(1,1,1) >>>(x);

  return 0;
}
```
