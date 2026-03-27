---
title: "3. Function Calling Sequence"
section: "3"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#function-calling-sequence"
---

# [3. Function Calling Sequence](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#function-calling-sequence)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#function-calling-sequence "Permalink to this headline")

This section describes the PTX-level function calling sequence, including register usage, stack-frame layout, and parameter passing. The PTX-level function calling sequence describes what gets represented in PTX to enable function calls. There is an abstraction at this level. Most of the details associated with the function calling sequence are handled at the SASS level.

PTX versions earlier than 2.0 do not conform to the ABI defined in this document, and cannot perform ABI compatible function calls. For the calling convention to work PTX version 2.0 or greater must be used.

## [3.1. Registers](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#registers)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#registers "Permalink to this headline")

At the PTX level, the registers that are specified are virtual. Register allocation occurs during PTX-to-SASS translation. The PTX-to-SASS translation also converts parameters and return values to physical registers or stack locations.

## [3.2. Stack Frame](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#stack-frame)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#stack-frame "Permalink to this headline")

The PTX level has no concept of the software stack. Manipulation of the stack is completely defined at the SASS level, and gets allocated during the PTX-to-SASS translation process.

## [3.3. Parameter Passing](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#parameter-passing)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#parameter-passing "Permalink to this headline")

At the PTX level, all parameters and return values present in a device function use the parameter state space (.param). The below table contains the rules for handling parameters and return values that are defined at the source level. For each source-level type, the corresponding PTX-level type that should be used is provided.

| Source Type | Size in Bits | PTX Type |
| --- | --- | --- |
| Integral types | 8 to 32 (A) | .u32 (if unsigned) or .s32 (if signed) |
| Integral types | 64 | .u64 (if unsigned) or .s64 (if signed) |
| Pointers (B) | 32 | .u32 |
| Pointers (B) | 64 | .u64 |
| Floating-point types (C) | 32 | .f32 |
| Floating-point types (C) | 64 | .f64 |
| Aggregates or unions | Any size | .align `align` .b8 `name`[`size`]   Where `align` is overall aggregate-or-union alignment in bytes (D), `name` is variable name associated with aggregate or union, and `size` is the aggregate-or-union size in bytes. |
| Handles (E) | 64 | .b64 (assigned from .texref, .sampleref, .surfref) |

NOTES:

1. Values shorter than 32-bits are sign extended or zero extended, depending on whether they are signed or unsigned types.
2. Unless the memory type is specified in the function declaration, all pointers passed at the PTX level must use a generic address.
3. 16-bit floating-point types are only used for storage. Therefore, they cannot be used for parameters or return values.
4. The alignment must be 1, 2, 4, 8, 16, 32, 64, or 128 bytes.
5. The PTX built-in opaque types such as texture, sampler, and surface types are can be passed into functions as parameters and be returned by them through 64-bit handles. The handle contains the necessary information to access the actual data from the texture or surface memory as well as the attributes of the object stored in its type descriptor. See section [Texture, Sampler, and Surface Types](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#textures-surfaces-samplers) for more information on handles.
