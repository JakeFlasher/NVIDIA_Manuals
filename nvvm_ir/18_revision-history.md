---
title: "18. Revision History"
section: "18"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#revision-history"
---

# [18. Revision History](https://docs.nvidia.com/cuda/nvvm-ir-spec#revision-history)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#revision-history "Permalink to this headline")

**Version 1.0**

- Initial Release.

**Version 1.1**

- Added support for UVM managed variables in global property annotation. See [Supported Properties](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#supported-properties).

**Version 1.2**

- Update to LLVM 3.4 for CUDA 7.0.
- Remove address space intrinsics in favor of `addrspacecast`.
- Add information about source level debugging support.

**Version 1.3**

- Add support for LLVM 3.8 for CUDA 8.0.

**Version 1.4**

- Add support for warp-level intrinsics.

**Version 1.5**

- Add support for LLVM 5.0 for CUDA 9.2.

**Version 1.6**

- Update to LLVM 7.0.1 for CUDA 11.2.

**Version 1.7**

- Add support for alloca with dynamic size.

**Version 1.8**

- Add support for i128 in data layout.

**Version 1.9**

- Modified text about ignoring shared variable initializations.

**Version 1.10**

- Added support for grid_constant kernel parameters for CUDA 11.7.

**Version 1.11**

- Added support for Hopper+ cluster intrinsics and max_blocks_per_cluster kernel property for CUDA 11.8.
- Deprecated support for 32-bit compilation.

**Version 2.0**

- Updated the NVVM IR to version 2.0 which is incompatible with NVVM IR version 1.x
- Removed address space conversion intrinsics. The IR verifier on 2.0 IR will give an error when these intrinsics are present. Clients of libNVVM are advised to use addrspacecast instruction instead.
- Stricter error checking on the supported datalayouts.
- Older style loop unroll pragma metadata on loop backedges is no longer supported. Clients are advised to use the newer loop pragma metadata defined by the LLVM framework.
- Shared variable initialization with non-undef values is no longer supported. In 1.x versions these initializers were ignored silently. This feature makes the 2.0 version incompatible with 1.x versions.
