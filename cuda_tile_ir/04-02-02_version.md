---
title: "4.2.2. Version"
section: "4.2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#version"
---

### [4.2.2. Version](https://docs.nvidia.com/cuda/tile-ir/latest/sections#version)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#version "Permalink to this headline")

Following the magic number is the version of the bytecode format. The version allows both
forwards and backwards compatibility of the format.

The version is encoded as three little-endian fixed-width fields immediately following the magic number:

```text
version {
  major: uint8_t,    // Major version number
  minor: uint8_t,    // Minor version number
  tag: uint16_t      // Version tag (little-endian)
}
```

Each file encodes the version which allows parsers to adapt to the specific version.
We do not break old versions of the format, if a newer file uses features an older parser
can’t interpret, it skips unknown optional features but will fail gracefully in cases where
new features are unsupported. A new producer can also target older versions of the bytecode
directly ensuring compatibility with older drivers.
