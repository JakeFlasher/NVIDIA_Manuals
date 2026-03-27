---
title: "13. Global Property Annotation"
section: "13"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#global-property-annotation"
---

# [13. Global Property Annotation](https://docs.nvidia.com/cuda/nvvm-ir-spec#global-property-annotation)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#global-property-annotation "Permalink to this headline")

## [13.1. Overview](https://docs.nvidia.com/cuda/nvvm-ir-spec#overview)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#overview "Permalink to this headline")

NVVM uses Named Metadata to annotate IR objects with properties that are otherwise not representable in the IR. The NVVM IR producers can use the Named Metadata to annotate the IR with properties, which the NVVM compiler can process.

## [13.2. Representation of Properties](https://docs.nvidia.com/cuda/nvvm-ir-spec#representation-of-properties)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#representation-of-properties "Permalink to this headline")

For each translation unit (that is, per bitcode file), there is a named metadata called `nvvm.annotations`.

This named metadata contains a list of MDNodes.

The first operand of each MDNode is an entity that the node is annotating using the remaining operands.

Multiple MDNodes may provide annotations for the same entity, in which case their first operands will be same.

The remaining operands of the MDNode are organized in order as <property-name, value>.

- The property-name operand is MDString, while the value is `i32`.
- Starting with the operand after the annotated entity, every alternate operand specifies a property.
- The operand after a property is its value.

The following is an example.

```text
!nvvm.annotations = !{!12, !13}
  !12 = !{void (i32, i32)* @_Z6kernelii, !"kernel", i32 1}
  !13 = !{void ()* @_Z7kernel2v, !"kernel", i32 1, !"maxntidx", i32 16}
```

If two bitcode files are being linked and both have a named metadata `nvvm.annotations`, the linked file will have a single merged named metadata. If both files define properties for the same entity foo , the linked file will have two MDNodes defining properties for foo . It is illegal for the files to have conflicting properties for the same entity.

## [13.3. Supported Properties](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-properties)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-properties "Permalink to this headline")

| Property Name | Annotated On | Description |
| --- | --- | --- |
| `maxntid{x, y, z}` | kernel function | Maximum expected CTA size from any launch. |
| `reqntid{x, y, z}` | kernel function | Minimum expected CTA size from any launch. |
| `cluster_dim_{x,y,z}` | kernel function | Support for cluster dimensions for Hopper+. If any dimension is specified as 0, then all dimensions must be specified as 0. |
| `cluster_max_blocks` | kernel function | Maximum number of blocks per cluster. Must be non-zero. Only supported for Hopper+. |
| `minctasm` | kernel function | Hint/directive to the compiler/driver, asking it to put at least these many CTAs on an SM. |
| `grid_constant` | kernel function | The argument is a metadata node, which contains a list of integers, where each integer n denotes that the nth parameter has the grid_constant annotation (numbering from 1). The parameter’s type must be of pointer type with byval attribute set. It is undefined behavior to write to memory pointed to by the parameter. This property is only supported for Volta+. |
| `maxnreg` | function | Maximum number of registers for function. |
| `kernel` | function | Signifies that this function is a kernel function. |
| `align` | function | Signifies that the value in low 16-bits of the 32-bit value contains alignment of n th parameter type if its alignment is not the natural alignment. n is specified by high 16-bits of the value. For return type, n is 0. |
| `texture` | global variable | Signifies that variable is a texture. |
| `surface` | global variable | Signifies that variable is a surface. |
| `managed` | global variable | Signifies that variable is a UVM managed variable. |
