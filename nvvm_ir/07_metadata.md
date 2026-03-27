---
title: "7. Metadata"
section: "7"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#metadata"
---

# [7. Metadata](https://docs.nvidia.com/cuda/nvvm-ir-spec#metadata)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#metadata "Permalink to this headline")

## [7.1. Metadata Nodes and Metadata Strings](https://docs.nvidia.com/cuda/nvvm-ir-spec#metadata-nodes-and-metadata-strings)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#metadata-nodes-and-metadata-strings "Permalink to this headline")

Supported.

The following metadata are understood by the NVVM compiler:

- Specialized Metadata Nodes
- `llvm.loop.unroll.count`
- `llvm.loop.unroll.disable`
- `llvm.loop.unroll.full`
- `callalign` (see [Rules and Restrictions](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#rules-and-restrictions) for Calling Conventions)

Module flags metadata (`llvm.module.flags`) is supported and verified, but the metadata values will be ignored.

All other metadata is accepted and ignored.
