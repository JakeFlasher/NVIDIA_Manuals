---
title: "3. High Level Structure"
section: "3"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#high-level-structure"
---

# [3. High Level Structure](https://docs.nvidia.com/cuda/nvvm-ir-spec#high-level-structure)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#high-level-structure "Permalink to this headline")

## [3.1. Linkage Types](https://docs.nvidia.com/cuda/nvvm-ir-spec#linkage-types)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#linkage-types "Permalink to this headline")

Supported:

- `private`
- `internal`
- `available_externally`
- `linkonce`
- `weak`
- `common`
- `linkonce_odr`
- `weak_odr`
- `external`

All other linkage types are not supported.

See [NVVM ABI for PTX](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-abi-for-ptx) for details on how linkage types are translated to PTX.

## [3.2. Calling Conventions](https://docs.nvidia.com/cuda/nvvm-ir-spec#calling-conventions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#calling-conventions "Permalink to this headline")

All LLVM calling convention markings are accepted and ignored. Functions and calls are generated according to the PTX calling convention.

### [3.2.1. Rules and Restrictions](https://docs.nvidia.com/cuda/nvvm-ir-spec#rules-and-restrictions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#rules-and-restrictions "Permalink to this headline")

1. When an argument with width less than 32-bit is passed, the `zeroext/signext` parameter attribute should be set. `zeroext` will be assumed if not set.
2. When a value with width less than 32-bit is returned, the `zeroext/signext` parameter attribute should be set. `zeroext` will be assumed if not set.
3. Arguments of aggregate or vector types that are passed by value can be passed by pointer with the `byval` attribute set (referred to as the `by-pointer-byval` case below). The align attribute must be set if the type requires a non-natural alignment (natural alignment is the alignment inferred for the aggregate type according to the [Data Layout](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout) section).
4. If a function has an argument of aggregate or vector type that is passed by value directly and the type has a non-natural alignment requirement, the alignment must be annotated by the global property annotation <`align`, alignment>, where alignment is a 32-bit integer whose upper 16 bits represent the argument position (starting from 1) and the lower 16 bits represent the alignment.
5. If the return type of a function is an aggregate or a vector that has a non-natural alignment, then the alignment requirement must be annotated by the global property annotation <`align`, alignment>, where the upper 16 bits is 0, and the lower 16 bits represent the alignment.
6. It is not required to annotate a function with <`align`, alignment> otherwise. If annotated, the alignment must match the natural alignment or the align attribute in the `by-pointer-byval` case.
7. For an indirect call instruction of a function that has a non-natural alignment for its return value or one of its arguments that is not expressed in alignment in the `by-pointer-byval` case, the call instruction must have an attached metadata of kind `callalign`. The metadata contains a sequence of `i32` fields each of which represents a non-natural alignment requirement. The upper 16 bits of an `i32` field represent the argument position (0 for return value, 1 for the first argument, and so on) and the lower 16 bits represent the alignment. The `i32` fields must be sorted in the increasing order.

For example,

```c
%call = call %struct.S %fp1(%struct.S* byval align 8 %arg1p, %struct.S %arg2),!callalign !10
!10 = !{i32 8, i32 520};
```
8. It is not required to have an `i32` metadata field for the other arguments or the return value otherwise. If presented, the alignment must match the natural alignment or the align attribute in the `by-pointer-byval case`.
9. It is not required to have a `callalign` metadata attached to a direct call instruction. If attached, the alignment must match the natural alignment or the alignment in the `by-pointer-byval` case.
10. The absence of the metadata in an indirect call instruction means using natural alignment or the align attribute in the `by-pointer-byval` case.

## [3.3. Visibility Styles](https://docs.nvidia.com/cuda/nvvm-ir-spec#visibility-styles)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#visibility-styles "Permalink to this headline")

All styles—default, hidden, and protected—are accepted and ignored.

## [3.4. DLL Storage Classes](https://docs.nvidia.com/cuda/nvvm-ir-spec#dll-storage-classes)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#dll-storage-classes "Permalink to this headline")

Not supported.

## [3.5. Thread Local Storage Models](https://docs.nvidia.com/cuda/nvvm-ir-spec#thread-local-storage-models)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#thread-local-storage-models "Permalink to this headline")

Not supported.

## [3.6. Runtime Preemption Specifiers](https://docs.nvidia.com/cuda/nvvm-ir-spec#runtime-preemption-specifiers)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#runtime-preemption-specifiers "Permalink to this headline")

Not supported.

## [3.7. Structure Types](https://docs.nvidia.com/cuda/nvvm-ir-spec#structure-types)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#structure-types "Permalink to this headline")

Supported.

## [3.8. Non-Integral Pointer Type](https://docs.nvidia.com/cuda/nvvm-ir-spec#non-integral-pointer-type)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#non-integral-pointer-type "Permalink to this headline")

Not supported.

## [3.9. Comdats](https://docs.nvidia.com/cuda/nvvm-ir-spec#comdats)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#comdats "Permalink to this headline")

Not supported.

## [3.10. source_filename](https://docs.nvidia.com/cuda/nvvm-ir-spec#source-filename)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#source-filename "Permalink to this headline")

Accepted and ignored.

## [3.11. Global Variables](https://docs.nvidia.com/cuda/nvvm-ir-spec#global-variables)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#global-variables "Permalink to this headline")

A global variable, that is not an intrinsic global variable, may be optionally declared to reside in one of the following address spaces:

- `global`
- `shared`
- `constant`

If no address space is explicitly specified, the global variable is assumed to reside in the `global` address space with a generic address value. See [Address Space](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#address-space) for details.

`thread_local` variables are not supported.

No explicit section (except for the metadata section) is allowed.

Initializations of `shared` variables are not supported. Use undef initialization.

## [3.12. Functions](https://docs.nvidia.com/cuda/nvvm-ir-spec#functions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#functions "Permalink to this headline")

The following are not supported on functions:

- Alignment
- Explicit section
- Garbage collector name
- Prefix data
- Prologue
- Personality

## [3.13. Aliases](https://docs.nvidia.com/cuda/nvvm-ir-spec#aliases)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#aliases "Permalink to this headline")

Supported only as aliases of non-kernel functions.

## [3.14. Ifuncs](https://docs.nvidia.com/cuda/nvvm-ir-spec#ifuncs)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#ifuncs "Permalink to this headline")

Not supported.

## [3.15. Named Metadata](https://docs.nvidia.com/cuda/nvvm-ir-spec#named-metadata)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#named-metadata "Permalink to this headline")

Accepted and ignored, except for the following:

- `!nvvm.annotations`: see [Global Property Annotation](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#global-property-annotation-chapter-11)
- `!nvvmir.version`
- `!llvm.dbg.cu`
- `!llvm.module.flags`

The NVVM IR version is specified using a named metadata called `!nvvmir.version`. The `!nvvmir.version` named metadata may have one metadata node that contains the NVVM IR version for that module. If multiple such modules are linked together, the named metadata in the linked module may have more than one metadata node with each node containing a version. A metadata node with NVVM IR version takes either of the following forms:

- It may consist of two i32 values—the first denotes the NVVM IR major version number and the second denotes the minor version number. If absent, the version number is assumed to be 1.0, which can be specified as:

```text
!nvvmir.version = !{!0}
!0 = !{i32 1, i32 0}
```
- It may consist of four i32 values—the first two denote the NVVM IR major and minor versions respectively. The third value denotes the NVVM IR debug metadata major version number, and the fourth value denotes the corresponding minor version number. If absent, the version number is assumed to be 1.0, which can be specified as:

```text
!nvvmir.version = !{!0}
!0 = !{i32 1, i32 0, i32 1, i32 0}
```

The version of NVVM IR described in this document is 2.0. The version of NVVM IR debug metadata described in this document is 3.1.

## [3.16. Parameter Attributes](https://docs.nvidia.com/cuda/nvvm-ir-spec#parameter-attributes)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#parameter-attributes "Permalink to this headline")

Supported, except the following:

Accepted and ignored:

- `inreg`
- `nest`

All other parameter attributes are not supported.

See [Calling Conventions](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#calling-conventions) for the use of the attributes.

## [3.17. Garbage Collector Strategy Names](https://docs.nvidia.com/cuda/nvvm-ir-spec#garbage-collector-strategy-names)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#garbage-collector-strategy-names "Permalink to this headline")

Not supported.

## [3.18. Prefix Data](https://docs.nvidia.com/cuda/nvvm-ir-spec#prefix-data)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#prefix-data "Permalink to this headline")

Not supported.

## [3.19. Prologue Data](https://docs.nvidia.com/cuda/nvvm-ir-spec#prologue-data)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#prologue-data "Permalink to this headline")

Not supported.

## [3.20. Attribute Groups](https://docs.nvidia.com/cuda/nvvm-ir-spec#attribute-groups)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#attribute-groups "Permalink to this headline")

Supported. The set of supported attributes is equal to the set of attributes accepted where the attribute group is used.

## [3.21. Function Attributes](https://docs.nvidia.com/cuda/nvvm-ir-spec#function-attributes)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#function-attributes "Permalink to this headline")

Supported:

- `allocsize`
- `alwaysinline`
- `cold`
- `convergent`
- `inaccessiblememonly`
- `inaccessiblemem_or_argmemonly`
- `inlinehint`
- `minsize`
- `no-jump-tables`
- `noduplicate`
- `noinline`
- `noreturn`
- `norecurse`
- `nounwind`
- `"null-pointer-is-valid"`
- `optforfuzzing`
- `optnone`
- `optsize`
- `readnone`
- `readonly`
- `writeonly`
- `argmemonly`
- `speculatable`
- `strictfp`

All other function attributes are not supported.

## [3.22. Global Attributes](https://docs.nvidia.com/cuda/nvvm-ir-spec#global-attributes)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#global-attributes "Permalink to this headline")

Not supported.

## [3.23. Operand Bundles](https://docs.nvidia.com/cuda/nvvm-ir-spec#operand-bundles)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#operand-bundles "Permalink to this headline")

Not supported.

## [3.24. Module-Level Inline Assembly](https://docs.nvidia.com/cuda/nvvm-ir-spec#module-level-inline-assembly)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#module-level-inline-assembly "Permalink to this headline")

Supported.

## [3.25. Data Layout](https://docs.nvidia.com/cuda/nvvm-ir-spec#data-layout)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#data-layout "Permalink to this headline")

Only the following data layout is supported:

- 64-bit

e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64

The following data layouts are deprecated and will be removed in a future release.

- 32-bit

e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64

e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64
- 64-bit

e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64

## [3.26. Target Triple](https://docs.nvidia.com/cuda/nvvm-ir-spec#target-triple)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#target-triple "Permalink to this headline")

Only the following target triple is supported, where * can be any name:

- 64-bit: `nvptx64-*-cuda`

The following target triple is deprecated, and will be removed in future release:

- 32-bit: `nvptx-*-cuda`

## [3.27. Pointer Aliasing Rules](https://docs.nvidia.com/cuda/nvvm-ir-spec#pointer-aliasing-rules)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#pointer-aliasing-rules "Permalink to this headline")

Supported.

## [3.28. Volatile Memory Access](https://docs.nvidia.com/cuda/nvvm-ir-spec#volatile-memory-access)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#volatile-memory-access "Permalink to this headline")

Supported. Note that for code generation: `ld.volatile` and `st.volatile` will be generated.

## [3.29. Memory Model for Concurrent Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#memory-model-for-concurrent-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#memory-model-for-concurrent-operations "Permalink to this headline")

Not applicable. Threads in an NVVM IR program must use atomic operations or barrier synchronization to communicate.

## [3.30. Atomic Memory Ordering Constraints](https://docs.nvidia.com/cuda/nvvm-ir-spec#atomic-memory-ordering-constraints)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#atomic-memory-ordering-constraints "Permalink to this headline")

Atomic loads and stores are not supported. Other atomic operations on other than 32-bit or 64-bit operands are not supported.

## [3.31. Fast-Math Flags](https://docs.nvidia.com/cuda/nvvm-ir-spec#fast-math-flags)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#fast-math-flags "Permalink to this headline")

Supported.

## [3.32. Use-list Order Directives](https://docs.nvidia.com/cuda/nvvm-ir-spec#use-list-order-directives)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#use-list-order-directives "Permalink to this headline")

Not supported.
