---
title: "warp submodule"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html#module-cutlass.cute.nvgpu.warp"
---

# [warp submodule](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.nvgpu.warp)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.nvgpu.warp "Permalink to this headline")

```
_`class`_`cutlass.cute.nvgpu.warp.``Field`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.Field "Link to this definition")
```

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

```
`ACCUMULATE`_`=` `'accum_c'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.Field.ACCUMULATE "Link to this definition")
```

```
`SFA`_`=` `'sf_a'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.Field.SFA "Link to this definition")
```

```
`SFB`_`=` `'sf_b'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.Field.SFB "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.warp.``MmaF16BF16Op`(
```

Bases: `WarpMmaOp`

F16/BF16 warp-level MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

```
`ab_dtype`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.MmaF16BF16Op.ab_dtype "Link to this definition")
```

```
`acc_dtype`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.MmaF16BF16Op.acc_dtype "Link to this definition")
```

```
`shape_mnk`_`:` `cutlass.cute.typing.Shape`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.MmaF16BF16Op.shape_mnk "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``MmaMXF4Op`(
```

Bases: `MmaSM120BlockScaledOp`

MXF4 warp-level MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
.kind           = {.kind::mxf4};
.scale_vec_size = {.scale_vec::2X};
.stype          = {.ue8m0};

```
`descriptive_name`_`=` `'warp-level` `MXF4` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.MmaMXF4Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``MmaMXF4NVF4Op`(
```

Bases: `MmaSM120BlockScaledOp`

MXF4NVF4 warp-level MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
.kind           = {.kind::mxf4nvf4};
.scale_vec_size = {.scale_vec::2X, .scale_vec::4X};
.stype          = {.ue8m0, .ue4m3};

```
`descriptive_name`_`=` `'warp-level` `MXF4NVF4` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``LdMatrix8x8x16bOp`(
```

Bases: `BaseOp`

8x8 `ldmatrix` Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix).
This operation corresponds to the `.m8n8` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``LdMatrix16x8x8bOp`(
```

Bases: `BaseOp`

16x8 8b `ldmatrix` Operation with transpose

There is no direct PTX correspondance to this Op.
This actually lowers to ldmatrix with the `.m16n16` qualifier and
additional address and value permutations to match stmatrix.m16n8.trans.
Useful for vectorizing with Ampere-style 8x8 matrix thread-value layouts

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``LdMatrix16x16x8bOp`(
```

Bases: `BaseOp`

16x16 `ldmatrix` Operation with transpose and optional unpacking to 8b container.
Packed source container is 16x4b elements with 64b padding
or 16x6b elements with 32b padding (total 128b per 16 elements)

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix).
This operation corresponds to the `.m16n16` and the `.b4x16_p64`,``.b6x16_p32``,``.b8`` qualifiers.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``StMatrix8x8x16bOp`(
```

Bases: `BaseOp`

8x8 `stmatrix` Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
This operation corresponds to the `m8n8` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warp.``StMatrix16x8x8bOp`(
```

Bases: `BaseOp`

16x8 `stmatrix` Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
This operation corresponds to the `m16n8` qualifier.

```
`__init__`(
```
