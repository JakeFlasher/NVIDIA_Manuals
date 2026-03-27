---
title: "Data Types"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#data-types"
---

## [Data Types](https://docs.nvidia.com/cuda/cutile-python#data-types)[](https://docs.nvidia.com/cuda/cutile-python/#data-types "Permalink to this headline")

```
_`class`_`cuda.tile.``DType`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.DType "Link to this definition")
```

A _data type_ (or _dtype_) describes the type of the objects of an [array](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array), [tile](https://docs.nvidia.com/cuda/cutile-python/#data-tiles-and-scalars), or
operation.

[Dtypes](https://docs.nvidia.com/cuda/cutile-python/#data-data-types) determine how values are stored in memory and how operations on those values are
performed.
[Dtypes](https://docs.nvidia.com/cuda/cutile-python/#data-data-types) are immutable.

[Dtypes](https://docs.nvidia.com/cuda/cutile-python/#data-data-types) can be used in [host code](https://docs.nvidia.com/cuda/cutile-python/execution.html#host-code) and [tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code).
They can be [kernel](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-tile-kernels) parameters.

```
_`property`_`bitwidth`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.DType.bitwidth "Link to this definition")
```

The number of bits in an element of the [data type](https://docs.nvidia.com/cuda/cutile-python/#data-data-types).

```
_`property`_`name`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.DType.name "Link to this definition")
```

The name of the [data type](https://docs.nvidia.com/cuda/cutile-python/#data-data-types).

```
`cuda.tile.``bool_`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.bool_ "Link to this definition")
```

A 8-bit [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) (`True` or `False`).

```
`cuda.tile.``uint8`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.uint8 "Link to this definition")
```

A 8-bit unsigned integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [0, +256].

```
`cuda.tile.``uint16`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.uint16 "Link to this definition")
```

A 16-bit unsigned integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [0, +65,536].

```
`cuda.tile.``uint32`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.uint32 "Link to this definition")
```

A 32-bit unsigned integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [0, +4,294,967,295].

```
`cuda.tile.``uint64`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.uint64 "Link to this definition")
```

A 64-bit unsigned integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [0, +18,446,744,073,709,551,615].

```
`cuda.tile.``int8`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.int8 "Link to this definition")
```

A 8-bit signed integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [−128, +127].

```
`cuda.tile.``int16`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.int16 "Link to this definition")
```

A 16-bit signed integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [−32,768, +32,767].

```
`cuda.tile.``int32`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.int32 "Link to this definition")
```

A 32-bit signed integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [−2,147,483,648, +2,147,483,647].

```
`cuda.tile.``int64`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.int64 "Link to this definition")
```

A 64-bit signed integer [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) whose values exist on the interval [−9,223,372,036,854,775,808, +9,223,372,036,854,775,807].

```
`cuda.tile.``float16`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.float16 "Link to this definition")
```

A IEEE 754 half-precision (16-bit) binary floating-point [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) (see [IEEE 754-2019](https://standards.ieee.org/standard/754-2019.html)).

```
`cuda.tile.``float32`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.float32 "Link to this definition")
```

A IEEE 754 single-precision (32-bit) binary floating-point [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) (see [IEEE 754-2019](https://standards.ieee.org/standard/754-2019.html)).

```
`cuda.tile.``float64`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.float64 "Link to this definition")
```

A IEEE 754 double-precision (64-bit) binary floating-point [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) (see [IEEE 754-2019](https://standards.ieee.org/standard/754-2019.html)).

```
`cuda.tile.``bfloat16`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.bfloat16 "Link to this definition")
```

A 16-bit floating-point [arithmetic dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) with 1 sign bit, 8 exponent bits, and 7 mantissa bits.

```
`cuda.tile.``tfloat32`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.tfloat32 "Link to this definition")
```

A 32-bit tensor floating-point [numeric dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) with 1 sign bit, 8 exponent bits, and 10 mantissa bits (19-bit representation stored in 32-bit container).

```
`cuda.tile.``float8_e4m3fn`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.float8_e4m3fn "Link to this definition")
```

An 8-bit floating-point [numeric dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) with 1 sign bit, 4 exponent bits, and 3 mantissa bits.

```
`cuda.tile.``float8_e5m2`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.float8_e5m2 "Link to this definition")
```

An 8-bit floating-point [numeric dtype](https://docs.nvidia.com/cuda/cutile-python/#data-numeric-arithmetic-data-types) with 1 sign bit, 5 exponent bits, and 2 mantissa bits.
