---
title: "warpgroup submodule"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warpgroup.html#module-cutlass.cute.nvgpu.warpgroup"
---

# [warpgroup submodule](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.nvgpu.warpgroup)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.nvgpu.warpgroup "Permalink to this headline")

```
_`class`_`cutlass.cute.nvgpu.warpgroup.``OperandMajorMode`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "Link to this definition")
```

Bases: `Enum`

An enumeration for the majorness of the input operands of the MMA.

```
_`class`_`cutlass.cute.nvgpu.warpgroup.``OperandSource`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.OperandSource "Link to this definition")
```

Bases: `Enum`

An enumeration for the source memory location of the A input operand of the MMA.

```
_`class`_`cutlass.cute.nvgpu.warpgroup.``Field`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.Field "Link to this definition")
```

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

```
`ACCUMULATE`_`=` `'accum_c'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.Field.ACCUMULATE "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.warpgroup.``MmaF16BF16Op`(
```

Bases: `MmaOp`

F16/BF16 warpgroup MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

```
`descriptive_name`_`=` `'warpgroup` `F16/BF16` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warpgroup.``MmaF8Op`(
```

Bases: `MmaOp`

F8 warpgroup MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

```
`descriptive_name`_`=` `'warpgroup` `F8` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.MmaF8Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.warpgroup.``SmemLayoutAtomKind`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "Link to this definition")
```

Bases: `Enum`

Enum class for the kinds of SMEM layout atoms for SM90.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can
be used to construct an SMEM layout using blocked product for operand A or B such that the
resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

```
`MN_INTER`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER "Link to this definition")
```

```
`MN_SW32`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32 "Link to this definition")
```

```
`MN_SW64`_`=` `3`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64 "Link to this definition")
```

```
`MN_SW128`_`=` `4`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128 "Link to this definition")
```

```
`K_INTER`_`=` `5`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER "Link to this definition")
```

```
`K_SW32`_`=` `6`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32 "Link to this definition")
```

```
`K_SW64`_`=` `7`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64 "Link to this definition")
```

```
`K_SW128`_`=` `8`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128 "Link to this definition")
```

```
`cutlass.cute.nvgpu.warpgroup.``make_smem_layout_atom`(
```

Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout
Atom kind and element data type.

**Parameters:**
: - **kind** ([_SmemLayoutAtomKind_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind")) – The kind of layout Atom
- **element_type** (_Type__[__Numeric__]_) – The element data type to construct the layout for

**Returns:**
: The SMEM layout atom

**Return type:**
: ComposedLayout

```
`cutlass.cute.nvgpu.warpgroup.``fence`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.fence "Link to this definition")
```

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-fence).

```
`cutlass.cute.nvgpu.warpgroup.``commit_group`(_`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.commit_group "Link to this definition")
```

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group).

```
`cutlass.cute.nvgpu.warpgroup.``wait_group`(_`group`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.warpgroup.wait_group "Link to this definition")
```

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group).
