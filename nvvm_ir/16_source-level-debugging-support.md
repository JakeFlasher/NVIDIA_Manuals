---
title: "16. Source Level Debugging Support"
section: "16"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#source-level-debugging-support"
---

# [16. Source Level Debugging Support](https://docs.nvidia.com/cuda/nvvm-ir-spec#source-level-debugging-support)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#source-level-debugging-support "Permalink to this headline")

To enable source level debugging of an IR module, NVVM IR supports debug
intrinsics and debug information descriptors to express the debugging
information. Debug information descriptors are represented using specialized
metadata nodes. The current NVVM IR debug metadata version is 3.1.

NVVM IR debugging support is based on that in LLVM 7.0.1 for pre-Blackwell
targets and LLVM 21.1.0 for Blackwell and later targets. For the complete
semantics of the IR, readers of this chapter should refer to the official LLVM
IR [Specialized Metadata Nodes](https://releases.llvm.org/7.0.1/docs/LangRef.html#specialized-metadata-nodes)
and the [Source Level Debugging](https://releases.llvm.org/7.0.1/docs/SourceLevelDebugging.html) documents.
Blackwell and later targets should refer to [this](https://releases.llvm.org/21.1.0/docs/LangRef.html#specialized-metadata-nodes) and [this](https://releases.llvm.org/21.1.0/docs/SourceLevelDebugging.html) document
respectively.

The following metadata nodes need to be present in the module when debugging support is requested:

- Named metadata node `!llvm.dbg.cu`
- Module flags metadata for `"Debug Info Version"` flag: The _behavior_ flag should be `Error`. The value of the flag should be `DEBUG_METADATA_VERSION`, which is 3.
- Named metadata `!nvvmir.version` containing a metadata node with the NVVM IR major and minor version values followed by the NVVM IR debug metadata major and minor version values. The current NVVM IR debug metadata version is 3.1.
- The debug resolution (e.g., full, line info only) is controlled by the DICompileUnit’s `emissionKind` field:
  - `FullDebug (value: 1)`: Generate symbolic debug and line information. This requires the libNVVM `-g` option to be specified at compile time.
  - `DebugDirectivesOnly (value: 3)`: Generate line information.

Source level debugging is supported only for a single debug compile unit. If there are multiple input NVVM IR modules, at most one module may have a single debug compile unit.
