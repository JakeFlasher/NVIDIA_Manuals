---
title: "tcgen05 submodule"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html#module-cutlass.cute.nvgpu.tcgen05"
---

# [tcgen05 submodule](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.nvgpu.tcgen05)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.nvgpu.tcgen05 "Permalink to this headline")

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Repetition`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition "Link to this definition")
```

Bases: `Enum`

An enumeration for the number of repetitions of a given TMEM copy within the instruction.

```
`x1`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x1 "Link to this definition")
```

```
`x2`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x2 "Link to this definition")
```

```
`x4`_`=` `4`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x4 "Link to this definition")
```

```
`x8`_`=` `8`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x8 "Link to this definition")
```

```
`x16`_`=` `16`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x16 "Link to this definition")
```

```
`x32`_`=` `32`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x32 "Link to this definition")
```

```
`x64`_`=` `64`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x64 "Link to this definition")
```

```
`x128`_`=` `128`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Repetition.x128 "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``TmemLoadRedOp`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp "Link to this definition")
```

Bases: `Enum`

An enumeration for the possible reduce operations for TMEM load operations.

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Pack`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Pack "Link to this definition")
```

Bases: `Enum`

An enumeration for the possible packing patterns for TMEM to RMEM copies.

```
`NONE`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Pack.NONE "Link to this definition")
```

```
`PACK_16b_IN_32b`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Pack.PACK_16b_IN_32b "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Unpack`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Unpack "Link to this definition")
```

Bases: `Enum`

An enumeration for the possible unpacking patterns for RMEM to TMEM copies.

```
`NONE`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Unpack.NONE "Link to this definition")
```

```
`UNPACK_32b_IN_16b`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Unpack.UNPACK_32b_IN_16b "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Ld16x64bOp`(
```

Bases: `_LdBase`

16x64b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x64b` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Ld16x128bOp`(
```

Bases: `_LdBase`

16x128b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x128b` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Ld16x256bOp`(
```

Bases: `_LdBase`

16x256b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x256b` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Ld16x32bx2Op`(
```

Bases: `_LdBase`

16x32bx2 TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x32bx2` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Ld32x32bOp`(
```

Bases: `_LdBase`

32x32b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.32x32` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``St16x64bOp`(
```

Bases: `_StBase`

16x64b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x64` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``St16x128bOp`(
```

Bases: `_StBase`

16x128b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x128` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``St16x256bOp`(
```

Bases: `_StBase`

16x256b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x256` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``St16x32bx2Op`(
```

Bases: `_StBase`

16x32x2b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x32x2` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``St32x32bOp`(
```

Bases: `_StBase`

32x32b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.32x32` qualifier.

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``OperandMajorMode`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "Link to this definition")
```

Bases: `Enum`

An enumeration for the majorness of the input operands of the MMA.

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``OperandSource`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.OperandSource "Link to this definition")
```

Bases: `Enum`

An enumeration for the source memory location of the A input operand of the MMA.

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``CtaGroup`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.CtaGroup "Link to this definition")
```

Bases: `Enum`

An enumeration for the `cta_group`  qualifier of the MMA.

```
`ONE`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.CtaGroup.ONE "Link to this definition")
```

```
`TWO`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.CtaGroup.TWO "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``Field`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Field "Link to this definition")
```

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

```
`NEGATE_A`_`=` `'neg_a'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Field.NEGATE_A "Link to this definition")
```

```
`NEGATE_B`_`=` `'neg_b'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Field.NEGATE_B "Link to this definition")
```

```
`ACCUMULATE`_`=` `'accum_c'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Field.ACCUMULATE "Link to this definition")
```

```
`SFA`_`=` `'sf_a'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Field.SFA "Link to this definition")
```

```
`SFB`_`=` `'sf_b'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.Field.SFB "Link to this definition")
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaTF32Op`(
```

Bases: `MmaOp`

TF32 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::tf32` qualifier.

```
`descriptive_name`_`=` `'tcgen05` `TF32` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaTF32Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaF16BF16Op`(
```

Bases: `MmaOp`

F16/BF16 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::f16` qualifier.

```
`descriptive_name`_`=` `'tcgen05` `F16/BF16` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaI8Op`(
```

Bases: `MmaOp`

I8 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::i8` qualifier.

```
`descriptive_name`_`=` `'tcgen05` `I8` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaI8Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaFP8Op`(
```

Bases: `MmaOp`

F8 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).

```
`descriptive_name`_`=` `'tcgen05` `F8` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaFP8Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaMXF8Op`(
```

Bases: `BlockScaledMmaOp`

MXF8 tcgen05 BlockScaled MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::mxf8f6f4` qualifier.

```
`descriptive_name`_`=` `'tcgen05` `MXF8` `BlockScaled` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaMXF4Op`(
```

Bases: `BlockScaledMmaOp`

MXF4 tcgen05 BlockScaled MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::mxf4` qualifier.

```
`descriptive_name`_`=` `'tcgen05` `MXF4` `BlockScaled` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``MmaMXF4NVF4Op`(
```

Bases: `BlockScaledMmaOp`

MXF4NVF4 tcgen05 BlockScaled MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::mxf4nvf4` qualifier.

```
`descriptive_name`_`=` `'tcgen05` `MXF4NVF4` `BlockScaled` `MMA` `Operation'`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.descriptive_name "Link to this definition")
```

```
`__init__`(
```

```
_`class`_`cutlass.cute.nvgpu.tcgen05.``SmemLayoutAtomKind`(_`value`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "Link to this definition")
```

Bases: `Enum`

Enum class for the kinds of SMEM layout atoms for SM100.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can be
used to construct an SMEM layout using blocked product for operand A or B such that the
resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

```
`MN_INTER`_`=` `1`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER "Link to this definition")
```

```
`MN_SW32`_`=` `2`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW32 "Link to this definition")
```

```
`MN_SW64`_`=` `3`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64 "Link to this definition")
```

```
`MN_SW128`_`=` `4`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128 "Link to this definition")
```

```
`MN_SW128_32B`_`=` `5`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128_32B "Link to this definition")
```

```
`K_INTER`_`=` `6`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER "Link to this definition")
```

```
`K_SW32`_`=` `7`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW32 "Link to this definition")
```

```
`K_SW64`_`=` `8`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64 "Link to this definition")
```

```
`K_SW128`_`=` `9`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW128 "Link to this definition")
```

```
`cutlass.cute.nvgpu.tcgen05.``make_smem_layout_atom`(
```

Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout
Atom kind and element data type.

**Parameters:**
: - **kind** ([_SmemLayoutAtomKind_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind")) – The kind of layout Atom
- **element_type** (_Type__[__Numeric__]_) – The element data type to construct the layout for

**Returns:**
: The SMEM layout atom

**Return type:**
: ComposedLayout

```
`cutlass.cute.nvgpu.tcgen05.``tile_to_mma_shape`(
```

Tiles a layout to an MMA shape.

```
`cutlass.cute.nvgpu.tcgen05.``commit`(
```

Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

**Parameters:**
: - **mbar_ptr** (_Pointer_) – A pointer to the mbarrier in SMEM
- **mask** (_Int_) – An optional multicast mask for the CTAs in the cluster to signal arrival to

```
`cutlass.cute.nvgpu.tcgen05.``is_tmem_load`(_`atom``:` [`CopyAtom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.is_tmem_load "Link to this definition")
```

Returns whether a CopyAtom instance is a TMEM load.

```
`cutlass.cute.nvgpu.tcgen05.``is_tmem_store`(_`atom``:` [`CopyAtom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.nvgpu.tcgen05.is_tmem_store "Link to this definition")
```

Returns whether a CopyAtom instance is a TMEM store.

```
`cutlass.cute.nvgpu.tcgen05.``get_tmem_copy_properties`(
```

Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions,
and whether packing/unpacking is used).

```
`cutlass.cute.nvgpu.tcgen05.``find_tmem_tensor_col_offset`(
```

Computes the TMEM column offset given a TMEM tensor.

**Parameters:**
: **tmem_tensor** (_Tensor_) – The TMEM tensor to use to compute the columns offset

**Returns:**
: The columns offset

**Return type:**
: Int

```
`cutlass.cute.nvgpu.tcgen05.``make_tmem_copy`(
```

Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

```
`cutlass.cute.nvgpu.tcgen05.``make_s2t_copy`(
```

Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

```
`cutlass.cute.nvgpu.tcgen05.``get_s2t_smem_desc_tensor`(
```

Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.

```
`cutlass.cute.nvgpu.tcgen05.``make_umma_smem_desc`(
```

Construct shared memory descriptor for UMMA.

The *make_umma_smem_desc* operation accepts an input cute.ptr (optionally a nextSrc
pointer for the second buffer in a circular buffer scheme), alongside a cute.layout
and a major attr, then constructs the shared memory descriptor and returns it.
The layout must be describing the buffer pointed to by the input pointer and the
iterator must carry valid swizzle information.

There are 5 supported swizzle variants:
- S<0, 4, 3> | SWIZZLE_NONE
- S<1, 4, 3> | SWIZZLE_32B
- S<2, 4, 3> | SWIZZLE_64B
- S<3, 4, 3> | SWIZZLE_128B
- S<2, 5, 2> | SWIZZLE_128B_BASE32B

The cute.ptr must carry shared address space and must be aligned to 16B.

**Parameters:**
: - **src** (_Pointer_) – The source pointer to shared memory
- **layout** (_Layout_) – The layout describing the buffer
- **major** (_str_) – The major mode attribute
- **next_src** (_Optional__[__Pointer__]_) – Optional next source pointer for circular buffer scheme

**Returns:**
: The shared memory descriptor

**Return type:**
: SmemDescType
