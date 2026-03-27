---
title: "1. Introduction"
section: "1"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#introduction"
---

# [1. Introduction](https://docs.nvidia.com/cuda/nvvm-ir-spec#introduction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#introduction "Permalink to this headline")

NVVM IR is a compiler IR (intermediate representation) based on the LLVM IR. The NVVM IR is designed to represent GPU compute kernels (for example, CUDA kernels). High-level language front-ends, like the CUDA C compiler front-end, can generate NVVM IR. The NVVM compiler (which is based on LLVM) generates PTX code from NVVM IR.

NVVM IR and NVVM compilers are mostly agnostic about the source language being used. The PTX codegen part of a NVVM compiler needs to know the source language because of the difference in DCI (driver/compiler interface).

NVVM IR is a binary format and is based on a subset of LLVM IR bitcode format. This document uses only human-readable form to describe NVVM IR.

Technically speaking, NVVM IR is LLVM IR with a set of rules, restrictions, and conventions, plus a set of supported intrinsic functions. A program specified in NVVM IR is always a legal LLVM program. A legal LLVM program may not be a legal NVVM program.

There are three levels of support for NVVM IR.

- Supported: The feature is fully supported. Most IR features should fall into this category.
- Accepted and ignored: The NVVM compiler will accept this IR feature, but will ignore the required semantics. This applies to some IR features that do not have meaningful semantics on GPUs and that can be ignored. Calling convention markings are an example.
- Illegal, not supported: The specified semantics is not supported, such as a `fence` instruction. Future versions of NVVM may either support or accept and ignore IRs that are illegal in the current version.

This document describes version 2.0 of the NVVM IR and version 3.1 of the NVVM
debug metadata (see [Source Level Debugging Support](https://docs.nvidia.com/cuda/nvvm-ir-spec/#source-level-debugging-support)).  The 2.0 version of
NVVM IR is incompatible with the previous version 1.11.  Linking of NVVM IR
Version 1.11 with 2.0 will result in compiler error.

NVVM IR can be in one of two dialects. The LLVM 7 dialect is based on LLVM
7.0.1. The modern dialect is based on a more recent public release version of
LLVM (LLVM 21.1.0). The modern dialect only supports Blackwell and later
architectures (compute capability `compute_100` or greater).  For the
complete semantics of the IR, readers of this document should refer to either
the official LLVM Language Reference Manual [version 7](https://releases.llvm.org/7.0.1/docs/LangRef.html) or [version 21](https://releases.llvm.org/21.1.0/docs/LangRef.html).  This document is
annotated with notes when differences between the two NVVM IR dialects are
important.
