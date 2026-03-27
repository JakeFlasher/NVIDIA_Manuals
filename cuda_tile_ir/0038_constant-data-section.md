---
title: "Constant Data Section"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#constant-data-section"
---

#### [Constant Data Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#constant-data-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#constant-data-section "Permalink to this headline")

The constant data section holds large constants (e.g., dense tensors, large arrays) separately from code.

The current implementation does not impose explicit size limits on individual constants.
Constants are stored using `uint64_t` offsets, allowing for very large constant data.
However, practical limits may be imposed by available memory and any downstream
compiler or runtime limitations (such as CUBIN generation constraints).

As we have done with the string section we move the constants into their own section to avoid bloating
the function table section and allow for the lazy loading of specific constants when needed.
Individual operations may reference constants by an index into this section.
The section is encoded with the total number of constants, followed by the start index of each
of the individual constants. The remaining encoding contains a single blob containing all the
constants concatenated together.

```text
constant {
  numConstants: varint,
  constantStartIndex: uint64_t[],
  constantData: byte[]
}
```
