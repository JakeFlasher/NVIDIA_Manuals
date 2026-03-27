---
title: "String Section"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#string-section"
---

#### [String Section](https://docs.nvidia.com/cuda/tile-ir/latest/sections#string-section)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#string-section "Permalink to this headline")

The string section holds all textual names used by the module to avoid repeating them inline.
The section is encoded with the total number of strings, followed by the start
index of each of the individual strings. The remaining encoding contains a single blob containing all the strings
concatenated together. This design allows loading a specific string without reading the whole section. Strings in
the bytecode are stored as raw byte sequences (in UTF-8 encoding) with an associated size. Finding the i-th
string involves jumping to `stringStartIndex[i]` and reading until the next offset or end of the blob.

```text
strings {
  numStrings: varint,
  stringStartIndex: uint32[],
  stringData: byte[]
}
```
