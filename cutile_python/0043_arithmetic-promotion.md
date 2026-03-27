---
title: "Arithmetic Promotion"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#arithmetic-promotion"
---

## [Arithmetic Promotion](https://docs.nvidia.com/cuda/cutile-python#arithmetic-promotion)[](https://docs.nvidia.com/cuda/cutile-python/#arithmetic-promotion "Permalink to this headline")

Binary operations can be performed on two [tile](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) or [scalar](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars) operands of different [numeric dtypes](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types).

When both operands are [loosely typed numeric constants](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-constant-expressions-objects), then the result is also
a loosely typed constant: for example, `5 + 7` is a loosely typed integral constant 12,
and `5 + 3.0` is a loosely typed floating-point constant 8.0.

If any of the operands is not a [loosely typed numeric constant](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-constant-expressions-objects), then both are _promoted_
to a common dtype using the following process:

- Each operand is classified into one of the three categories:
_boolean_, _integral_, or _floating-point_.
The categories are ordered as follows: _boolean_ < _integral_ < _floating-point_.
- If either operand is a [loosely typed numeric constant](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-constant-expressions-objects), a concrete dtype is picked for it:
integral constants are treated as *int32*, *int64*, or *uint64*, depending on the value;
floating-point constants are treated as *float32*.
- If one of the two operands has a higher category than the other, then its concrete dtype
is chosen as the common dtype.
- If both operands are of the same category, but one of them is a [loosely typed numeric constant](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-constant-expressions-objects),
then the other operand’s dtype is picked as the common dtype.
- Otherwise, the common dtype is computed according to the table below.

|  | b1 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | tf32 | f8e4m3fn | f8e5m2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1 | b1 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u8 | u8 | u8 | u16 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u16 | u16 | u16 | u16 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u32 | u32 | u32 | u32 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| u64 | u64 | u64 | u64 | u64 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i8 | i8 | ERR | ERR | ERR | ERR | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i16 | i16 | ERR | ERR | ERR | ERR | i16 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i32 | i32 | ERR | ERR | ERR | ERR | i32 | i32 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| i64 | i64 | ERR | ERR | ERR | ERR | i64 | i64 | i64 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR |
| f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f32 | f64 | ERR | ERR | ERR | ERR |
| f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f64 | f32 | ERR | ERR | ERR |
| f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | ERR | ERR | ERR |
| bf | bf | bf | bf | bf | bf | bf | bf | bf | bf | ERR | f32 | f64 | bf | ERR | ERR | ERR |
| tf32 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | tf32 | ERR | ERR |
| f8e4m3fn | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e4m3fn | ERR |
| f8e5m2 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e5m2 |

Legend:

- b1: `bool_`
- u8: `uint8`
- u16: `uint16`
- u32: `uint32`
- u64: `uint64`
- i8: `int8`
- i16: `int16`
- i32: `int32`
- i64: `int64`
- f16: `float16`
- f32: `float32`
- f64: `float64`
- bf: `bfloat16`
- tf32: `tfloat32`
- f8e4m3fn: `float8_e4m3fn`
- f8e5m2: `float8_e5m2`
- f8e8m0fnu: `float8_e8m0fnu`
- f4e2m1fn: `float4_e2m1fn`
- ERR: Implicit promotion between these types is not supported
